"""
probe.py — Hallucination probe: stacking ensemble of four specialists.

Feature vector layout expected from aggregation.py (USE_GEOMETRIC=True):
  Block A  [0    :4480]  semantic  — mean-pooled hidden states, 5 layers × 896
  Block B  [4480 :4504]  drift     — inter-layer cosine similarities (24)
  Block C  [4504 :4529]  norms     — layer-wise mean L2 norms (25)
  Block D  [4529 :4534]  scalars   — norm_mean, norm_cv, norm_ratio, min_cos, n_real

Architecture (USE_GEOMETRIC=True):
  Specialist A   — PCA(64) on Block A       → LogisticRegression
  Specialist BC  — StandardScaler on B+C    → LogisticRegression
  Specialist D   — StandardScaler on Block D → LogisticRegression
  Specialist ALL — PCA(128) on all blocks   → LogisticRegression (C=0.1)

  Training:  5-fold OOF produces (N, 4) meta-features for the meta-learner.
             All specialists are then retrained on the full training set.
  Inference: specialists → 4 probabilities → meta LogisticRegression → label.

Fallback (USE_GEOMETRIC=False, feature_dim=4480):
  PCA(64) on Block A → LogisticRegression.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Block boundaries (must match aggregation.py) ────────────────────────────
_A_DIM  = 5 * 896   # 4480  Block A
_B_DIM  = 24        #        Block B
_C_DIM  = 25        #        Block C
_D_DIM  = 5         #        Block D

_A_END  = _A_DIM                        # 4480
_BC_END = _A_END  + _B_DIM + _C_DIM    # 4529
_D_END  = _BC_END + _D_DIM             # 4534

_N_META = 4         # number of specialist outputs fed to meta-learner


def _make_lr(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)


class HallucinationProbe(nn.Module):
    """Stacking ensemble for hallucination detection from hidden-state features."""

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5
        self._mode: str = "uninit"   # "stacking" | "fallback"

        # ── Preprocessors (fitted in fit()) ─────────────────────────────────
        self._ss_a    = StandardScaler()
        self._pca_a   = PCA(n_components=64,  random_state=42)
        self._ss_bc   = StandardScaler()
        self._ss_d    = StandardScaler()
        self._ss_all  = StandardScaler()
        self._pca_all = PCA(n_components=128, random_state=42)

        # ── Specialists (fitted on full training set after OOF) ──────────────
        self._spec_a   = _make_lr(C=1.0)
        self._spec_bc  = _make_lr(C=1.0)
        self._spec_d   = _make_lr(C=1.0)
        self._spec_all = _make_lr(C=0.1)

        # ── Meta-learner (fitted on OOF meta-features) ───────────────────────
        self._meta = _make_lr(C=1.0)

        # ── Fallback for USE_GEOMETRIC=False ─────────────────────────────────
        self._ss_fb  = StandardScaler()
        self._pca_fb = PCA(n_components=64, random_state=42)
        self._fb     = _make_lr(C=1.0)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _has_geo(X: np.ndarray) -> bool:
        return X.shape[1] > _A_END

    @staticmethod
    def _split(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (block_a, block_bc, block_d) views."""
        return X[:, :_A_END], X[:, _A_END:_BC_END], X[:, _BC_END:_D_END]

    def _transform(self, X: np.ndarray, fit: bool = False) -> tuple:
        """Scale + PCA each block; optionally fit the transformers."""
        a, bc, d = self._split(X)
        if fit:
            a_t   = self._pca_a.fit_transform(self._ss_a.fit_transform(a))
            bc_t  = self._ss_bc.fit_transform(bc)
            d_t   = self._ss_d.fit_transform(d)
            all_t = self._pca_all.fit_transform(self._ss_all.fit_transform(X))
        else:
            a_t   = self._pca_a.transform(self._ss_a.transform(a))
            bc_t  = self._ss_bc.transform(bc)
            d_t   = self._ss_d.transform(d)
            all_t = self._pca_all.transform(self._ss_all.transform(X))
        return a_t, bc_t, d_t, all_t

    def _meta_features(
        self,
        a_t: np.ndarray,
        bc_t: np.ndarray,
        d_t: np.ndarray,
        all_t: np.ndarray,
        specs: tuple,
    ) -> np.ndarray:
        """Concatenate specialist probabilities into meta-feature matrix."""
        sa, sbc, sd, sall = specs
        return np.column_stack([
            sa.predict_proba(a_t)[:, 1],
            sbc.predict_proba(bc_t)[:, 1],
            sd.predict_proba(d_t)[:, 1],
            sall.predict_proba(all_t)[:, 1],
        ])

    # ── Public interface ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Call predict() or predict_proba() directly.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the stacking ensemble on labelled feature vectors."""
        if not self._has_geo(X):
            self._fit_fallback(X, y)
            return self

        # ── Step 1: fit global preprocessors ────────────────────────────────
        self._transform(X, fit=True)

        # ── Step 2: generate OOF meta-features via internal 5-fold CV ───────
        n = len(y)
        oof_meta = np.zeros((n, _N_META))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for tr_idx, val_idx in skf.split(np.arange(n), y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr = y[tr_idx]

            # Local preprocessors for this fold
            a_tr, bc_tr, d_tr = self._split(X_tr)
            a_vl, bc_vl, d_vl = self._split(X_val)

            n_comp_a   = min(64,  len(y_tr) - 1, _A_DIM)
            n_comp_all = min(128, len(y_tr) - 1, X_tr.shape[1])

            ss_a, pca_a  = StandardScaler(), PCA(n_comp_a,   random_state=42)
            ss_bc        = StandardScaler()
            ss_d         = StandardScaler()
            ss_all, pca_all = StandardScaler(), PCA(n_comp_all, random_state=42)

            a_tr_t   = pca_a.fit_transform(ss_a.fit_transform(a_tr))
            a_vl_t   = pca_a.transform(ss_a.transform(a_vl))
            bc_tr_t  = ss_bc.fit_transform(bc_tr);  bc_vl_t  = ss_bc.transform(bc_vl)
            d_tr_t   = ss_d.fit_transform(d_tr);    d_vl_t   = ss_d.transform(d_vl)
            all_tr_t = pca_all.fit_transform(ss_all.fit_transform(X_tr))
            all_vl_t = pca_all.transform(ss_all.transform(X_val))

            # Fit fold specialists
            sa   = _make_lr(C=1.0).fit(a_tr_t,   y_tr)
            sbc  = _make_lr(C=1.0).fit(bc_tr_t,  y_tr)
            sd   = _make_lr(C=1.0).fit(d_tr_t,   y_tr)
            sall = _make_lr(C=0.1).fit(all_tr_t, y_tr)

            oof_meta[val_idx] = self._meta_features(
                a_vl_t, bc_vl_t, d_vl_t, all_vl_t, (sa, sbc, sd, sall)
            )

        # ── Step 3: train meta-learner on OOF meta-features ─────────────────
        self._meta.fit(oof_meta, y)

        # ── Step 4: retrain specialists on full training set ─────────────────
        a_t, bc_t, d_t, all_t = self._transform(X, fit=False)
        self._spec_a.fit(a_t,   y)
        self._spec_bc.fit(bc_t, y)
        self._spec_d.fit(d_t,   y)
        self._spec_all.fit(all_t, y)

        self._mode = "stacking"
        return self

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        n_comp = min(64, X.shape[0] - 1, X.shape[1])
        self._pca_fb = PCA(n_components=n_comp, random_state=42)
        X_t = self._pca_fb.fit_transform(self._ss_fb.fit_transform(X))
        self._fb.fit(X_t, y)
        self._mode = "fallback"

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune decision threshold on validation set to maximise F1."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            f1 = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self._threshold = best_t
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 2) probability array."""
        if self._mode == "fallback":
            X_t = self._pca_fb.transform(self._ss_fb.transform(X))
            return self._fb.predict_proba(X_t)

        a_t, bc_t, d_t, all_t = self._transform(X, fit=False)
        specs = (self._spec_a, self._spec_bc, self._spec_d, self._spec_all)
        meta_X = self._meta_features(a_t, bc_t, d_t, all_t, specs)
        return self._meta.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels using the tuned threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
