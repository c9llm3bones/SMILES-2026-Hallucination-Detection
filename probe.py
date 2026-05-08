"""
probe.py — Hallucination probe: PCA + LogisticRegression with C-tuning.

Feature vector layout expected from aggregation.py:
  Block A  [0    :4480]  semantic  — last real token, 5 layers × 896-d
  Block BC [4480 :4529]  geometric — inter-layer cosine (24) + layer norms (25)
  Block D  [4529 :4534]  scalars   — norm_mean, norm_cv, norm_ratio, min_cos, n_real

Pipeline (USE_GEOMETRIC=True):
  Block A  → StandardScaler → PCA(n_components=128)  ]
  Block BC → StandardScaler                           ] → concatenate → LogisticRegression(C=C*)
  Block D  → StandardScaler                           ]

  C* is selected from {0.01, 0.05, 0.1, 0.5, 1.0} by internal 3-fold CV (AUROC).

Fallback (USE_GEOMETRIC=False, feature_dim=4480):
  Block A → StandardScaler → PCA(64) → LogisticRegression(C=C*)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Block A boundary: 5 layers × 896-d = 4480 (must match aggregation.py)
_A_END = 25 * 896

_C_GRID = [0.01, 0.05, 0.1, 0.5, 1.0]


def _make_lr(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)


class HallucinationProbe(nn.Module):
    """Binary probe: per-block scaling + PCA on Block A + LogReg with C-tuning."""

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5
        self._has_geo: bool = False

        self._ss_a   = StandardScaler()
        self._pca_a  = PCA(n_components=128, random_state=42)
        self._ss_geo = StandardScaler()   # for Block BC + D concatenated
        self._model  = _make_lr(0.1)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _prepare(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale + PCA each block, return concatenated features."""
        a = X[:, :_A_END]

        n_comp = min(128, X.shape[0] - 1, _A_END)
        if fit:
            self._pca_a = PCA(n_components=n_comp, random_state=42)
            a_t = self._pca_a.fit_transform(self._ss_a.fit_transform(a))
        else:
            a_t = self._pca_a.transform(self._ss_a.transform(a))

        if not self._has_geo or X.shape[1] <= _A_END:
            return a_t

        geo = X[:, _A_END:]   # Block BC + D = 54-d
        if fit:
            geo_t = self._ss_geo.fit_transform(geo)
        else:
            geo_t = self._ss_geo.transform(geo)

        return np.hstack([a_t, geo_t])

    def _tune_c(self, X: np.ndarray, y: np.ndarray) -> float:
        """Pick C that maximises mean AUROC on internal 3-fold CV."""
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_c, best_auc = _C_GRID[0], -1.0
        for c in _C_GRID:
            aucs = []
            for tr, vl in skf.split(X, y):
                m = _make_lr(c)
                m.fit(X[tr], y[tr])
                try:
                    aucs.append(roc_auc_score(y[vl], m.predict_proba(X[vl])[:, 1]))
                except ValueError:
                    aucs.append(0.5)
            mean_auc = float(np.mean(aucs))
            if mean_auc > best_auc:
                best_auc, best_c = mean_auc, c
        return best_c

    # ── Public interface ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use predict() or predict_proba() directly.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self._has_geo = X.shape[1] > _A_END

        X_t = self._prepare(X, fit=True)
        best_c = self._tune_c(X_t, y)
        self._model = _make_lr(best_c)
        self._model.fit(X_t, y)
        return self

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
        """Return (n_samples, 2) class probability array."""
        return self._model.predict_proba(self._prepare(X, fit=False))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels using the tuned threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
