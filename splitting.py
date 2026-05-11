"""
splitting.py — RepeatedStratifiedKFold (5 splits × 5 repeats = 25 folds).

25-fold repeated CV gives a stable accuracy estimate that is independent of any
single lucky seed — a key advantage over single-split or plain 5-fold evaluation.

Split sizes (approximate, for n=689):
  test  : ~138 samples (20 %, 1/5)
  val   : ~83  samples (15 % of remaining ~551)
  train : ~468 samples
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """25-fold repeated stratified split returning (idx_train, idx_val, idx_test).

    Args:
        y:            Label array of shape (N,).
        df:           Unused; kept for interface compatibility.
        test_size:    Ignored (test size is fixed at 1/n_splits ≈ 20 %).
        val_size:     Fraction of non-test samples reserved for validation.
        random_state: Random seed passed to RepeatedStratifiedKFold.

    Returns:
        List of 25 (idx_train, idx_val, idx_test) tuples.
    """
    idx  = np.arange(len(y))
    rskf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=random_state
    )
    splits = []

    for train_val_idx, test_idx in rskf.split(idx, y):
        relative_val = val_size / (1.0 - 1.0 / 5)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[train_val_idx],
        )
        splits.append((train_idx, val_idx, test_idx))

    return splits
