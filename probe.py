"""
probe.py — Bootstrap ensemble of L2-regularised LogisticRegression probes.

Pipeline per fold:
  StandardScaler → 30 × LogisticRegression (bootstrap, C selected by 3-fold CV)

C is selected from {0.003, 0.005, 0.01, 0.05} by mean AUROC on internal 3-fold CV.
Final probability = mean over bootstrap models.  Threshold fixed at 0.5.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_N_BOOTSTRAP = 30
_C_GRID      = [0.001, 0.002, 0.003, 0.005]


def _make_lr(C: float) -> LogisticRegression:
    return LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42
    )


class HallucinationProbe(nn.Module):
    """Bootstrap ensemble probe — no PCA, no threshold tuning."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler:    StandardScaler             = StandardScaler()
        self._models:    list[LogisticRegression]   = []
        self._threshold: float                      = 0.5

    # ── Internal helpers ────────────────────────────────────────────────────

    def _tune_c(self, X: np.ndarray, y: np.ndarray) -> float:
        """Pick C maximising mean AUROC on internal 3-fold CV."""
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

    # ── Public interface ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use predict() or predict_proba() directly.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_s = self._scaler.fit_transform(X)
        best_c = self._tune_c(X_s, y)

        rng = np.random.default_rng(42)
        n   = len(y)
        self._models = []
        for _ in range(_N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            m   = _make_lr(best_c)
            m.fit(X_s[idx], y[idx])
            self._models.append(m)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        # Class prior 70/30 is informative; 0.5 threshold is reliable.
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_s   = self._scaler.transform(X)
        probs = np.mean(
            [m.predict_proba(X_s)[:, 1] for m in self._models], axis=0
        )
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
