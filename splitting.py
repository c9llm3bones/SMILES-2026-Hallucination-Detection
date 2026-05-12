"""
splitting.py — Train/val/test split strategy controlled by config.py.

n_repeats=1  → StratifiedKFold (single run)
n_repeats>1  → RepeatedStratifiedKFold (n_splits × n_repeats folds total)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from config import CFG

_SPLIT = CFG["splitting"]


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,   # unused — test size is fixed at 1/n_splits by KFold
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    n_splits  = _SPLIT["n_splits"]
    n_repeats = _SPLIT["n_repeats"]

    idx = np.arange(len(y))

    if n_repeats > 1:
        kfold = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    relative_val = val_size / (1.0 - 1.0 / n_splits)
    splits = []

    for train_val_idx, test_idx in kfold.split(idx, y):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[train_val_idx],
        )
        splits.append((train_idx, val_idx, test_idx))

    return splits
