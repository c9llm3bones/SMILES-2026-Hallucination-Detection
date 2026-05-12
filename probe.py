"""
probe.py — Hallucination probe controlled by config.py.

Pipeline:
  [StandardScaler] → [PCA] → bootstrap LogisticRegression ensemble
  Each option is enabled/disabled via config.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import CFG

_PROBE       = CFG["probe"]
_N_BOOTSTRAP = _PROBE["n_bootstrap"]
_C_GRID      = _PROBE["c_grid"]
_USE_PCA     = _PROBE["use_pca"]
_PCA_DIM     = _PROBE.get("pca_components")
_SCALE       = _PROBE["scale"]


def _make_lr(C: float) -> LogisticRegression:
    return LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42
    )


class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._scaler:    StandardScaler | None      = StandardScaler() if _SCALE else None
        self._pca:       PCA | None                 = None
        self._models:    list[LogisticRegression]   = []
        self._threshold: float                      = 0.5

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.fit_transform(X) if fit else self._scaler.transform(X)
        if _USE_PCA:
            if fit:
                n_comp = min(_PCA_DIM or 128, X.shape[0] - 1, X.shape[1])
                self._pca = PCA(n_components=n_comp, random_state=42)
                X = self._pca.fit_transform(X)
            else:
                X = self._pca.transform(X)
        return X

    def _tune_c(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(_C_GRID) == 1:
            return _C_GRID[0]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_c, best_auc = _C_GRID[0], -1.0
        for c in _C_GRID:
            aucs: list[float] = []
            for tr, vl in skf.split(X, y):
                m = _make_lr(c)
                m.fit(X[tr], y[tr])
                try:
                    aucs.append(roc_auc_score(y[vl], m.predict_proba(X[vl])[:, 1]))
                except ValueError:
                    aucs.append(0.5)
            if np.mean(aucs) > best_auc:
                best_auc, best_c = float(np.mean(aucs)), c
        return best_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_t    = self._transform(X, fit=True)
        best_c = self._tune_c(X_t, y)
        rng    = np.random.default_rng(42)
        n      = len(y)
        self._models = []
        for _ in range(_N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            m   = _make_lr(best_c)
            m.fit(X_t[idx], y[idx])
            self._models.append(m)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_t   = self._transform(X, fit=False)
        probs = np.mean(
            [m.predict_proba(X_t)[:, 1] for m in self._models], axis=0
        )
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
