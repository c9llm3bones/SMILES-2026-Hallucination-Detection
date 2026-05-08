"""
splitting.py — Train / validation / test split strategy.

Uses StratifiedKFold(n_splits=5) to maximise use of the 689-sample dataset.
Each fold preserves class balance in every subset.

Split sizes (approximate, for n=689, 70/30 imbalance):
  train : ~440 samples (64%)
  val   : ~110 samples (16%)
  test  : ~138 samples (20%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """5-fold stratified split returning (idx_train, idx_val, idx_test) tuples.

    The outer split (KFold) carves out the test set; the remaining samples are
    further split into train and val for threshold tuning.

    Args:
        y:            Label array of shape (N,).
        df:           Unused; kept for interface compatibility.
        test_size:    Approximate fraction for the test set per fold (~0.20
                      when n_splits=5, regardless of this parameter).
        val_size:     Fraction of non-test samples reserved for validation.
        random_state: Random seed.

    Returns:
        List of 5 (idx_train, idx_val, idx_test) tuples.
    """
    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []

    for train_val_idx, test_idx in skf.split(idx, y):
        # val_fraction relative to the train+val pool
        relative_val = val_size / (1.0 - 1.0 / 5)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[train_val_idx],
        )
        splits.append((train_idx, val_idx, test_idx))

    return splits
