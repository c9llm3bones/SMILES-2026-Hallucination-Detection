"""
config.py — Experiment configuration.

Set ACTIVE to reproduce any experiment:
  "baseline"  mean-pool all layers, LogReg C=1 no scale, 1×5-fold
  "sol2"      last-token L8/12/16/20/24, PCA(50)+bootstrap LR, 1×5-fold
  "sol3"      last-token L20–24, PCA(50)+bootstrap LR, 1×5-fold
  "sol4"      response-only max+mean pool L12–15, bootstrap LR, 25-fold  ← default

Note: Sol. 1 (stacking meta-learner) is not reproducible via this config.
"""

ACTIVE = "sol4"

CONFIGS: dict = {
    "baseline": {
        "aggregation": {
            "strategy": "mean_pool",
            "layers": list(range(25)),
            "response_only": False,
        },
        "probe": {
            "n_bootstrap": 1,
            "c_grid": [1.0],
            "use_pca": False,
            "pca_components": None,
            "scale": False,
        },
        "splitting": {
            "n_splits": 5,
            "n_repeats": 1,
        },
    },
    "sol2": {
        "aggregation": {
            "strategy": "last_token",
            "layers": [8, 12, 16, 20, 24],
            "response_only": False,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.01, 0.05, 0.1, 0.5, 1.0],
            "use_pca": True,
            "pca_components": 50,
            "scale": True,
        },
        "splitting": {
            "n_splits": 5,
            "n_repeats": 1,
        },
    },
    "sol3": {
        "aggregation": {
            "strategy": "last_token",
            "layers": [20, 21, 22, 23, 24],
            "response_only": False,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.01, 0.05, 0.1, 0.5, 1.0],
            "use_pca": True,
            "pca_components": 50,
            "scale": True,
        },
        "splitting": {
            "n_splits": 5,
            "n_repeats": 1,
        },
    },
    # ── Ablations (all identical to sol4 except one variable) ─────────────────

    # A0: prompt-only — completes the token-scope trio (prompt / response / all).
    "sol4_prompt_only": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [12, 13],
            "mean_pool_layers": [14, 15],
            "response_only": False,
            "prompt_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    # A1: all tokens vs response-only — isolates response_only effect.
    "sol4_all_tokens": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [12, 13],
            "mean_pool_layers": [14, 15],
            "response_only": False,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {
            "n_splits": 5,
            "n_repeats": 5,
        },
    },
    # A2: early layers — isolates layer selection effect.
    "sol4_early_layers": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [4, 5],
            "mean_pool_layers": [6, 7],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    # A3: late layers — isolates layer selection effect.
    "sol4_late_layers": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [20, 21],
            "mean_pool_layers": [22, 23],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    # A4: with PCA(50) — isolates PCA effect.
    "sol4_with_pca": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [12, 13],
            "mean_pool_layers": [14, 15],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": True,
            "pca_components": 50,
            "scale": True,
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    # A5: no bootstrap — isolates bootstrap ensemble effect.
    "sol4_no_bootstrap": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [12, 13],
            "mean_pool_layers": [14, 15],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 1,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    # A6: class_weight='balanced' — isolates class balancing effect.
    "sol4_balanced": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [12, 13],
            "mean_pool_layers": [14, 15],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
            "class_weight": "balanced",
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    # A7: max-pool only — isolates pooling type effect.
    "sol4_max_only": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [12, 13, 14, 15],
            "mean_pool_layers": [],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    # A8: mean-pool only — isolates pooling type effect.
    "sol4_mean_only": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [],
            "mean_pool_layers": [12, 13, 14, 15],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {"n_splits": 5, "n_repeats": 5},
    },
    "sol4": {
        "aggregation": {
            "strategy": "response_pool",
            "max_pool_layers": [12, 13],
            "mean_pool_layers": [14, 15],
            "response_only": True,
        },
        "probe": {
            "n_bootstrap": 30,
            "c_grid": [0.001, 0.002, 0.003, 0.005],
            "use_pca": False,
            "pca_components": None,
            "scale": True,
        },
        "splitting": {
            "n_splits": 5,
            "n_repeats": 5,
        },
    },
}

CFG = CONFIGS[ACTIVE]
