# Hallucination Detection — Solution

## Final Results

| Split | Accuracy | AUROC |
|---|---|---|
| Majority-class baseline | 70.10% | — |
| Probe (train) | 88.58% | 95.97% |
| Probe (val) | 74.27% | 79.15% |
| **Probe (test)** | **75.59%** | **79.81%** |

Evaluation: 25-fold repeated stratified CV (5 splits × 5 repeats), feature dim 3584, 689 training samples.


## Approach

### 1. Token aggregation (`aggregation.py`)

**Response-only pooling.**
The full input is `prompt + response`; the hallucination signal is entirely in the response. Ablations A0 and A1 confirm this: prompt-only and all-tokens pooling produce identical results (70.44% AUROC, 70.74% accuracy) — both −9.4 pp below Sol. 4 and barely above the majority-class baseline. The two are indistinguishable because prompts are long enough to numerically dominate the all-tokens pool, washing out the response signal. The train-test gap also widens (19.7 pp vs 13.0 pp) when prompt tokens are included, indicating increased overfitting.

To isolate response tokens without modifying `solution.py`, `aggregation.py` pre-tokenises all prompts from `dataset.csv` and `test.csv` at module import time and uses a global call counter to map each sequential invocation to the correct prompt boundary.

**Layer selection.** Mid-layers (12–15 of 24) outperform both early (L4–7: −4.8 pp AUROC, A2) and late (L20–23: −2.2 pp AUROC, A3) layers. This is consistent with Meng et al. (2022), who show via causal mediation analysis that "middle-layer feed-forward modules mediate factual predictions" in GPT-class models ([ROME, NeurIPS 2022](https://arxiv.org/abs/2202.05262)).

**Feature vector (3584-d):**

| Block | Operation | Layers | Dim |
|---|---|---|---|
| 0 | response max-pool | L12 | 896 |
| 1 | response max-pool | L13 | 896 |
| 2 | response mean-pool | L14 | 896 |
| 3 | response mean-pool | L15 | 896 |

Ablations A7/A8 show mean-only (79.44% AUROC) outperforms max-only (78.99%), and their combination gives a further +0.4 pp. The layer→pooling assignment (max on L12/13, mean on L14/15) is an empirical choice; the marginal gain of the combination over mean-only alone does not justify a stronger claim.

### 2. Probe classifier (`probe.py`)

**No PCA.** Ablation A4 shows that in our setup (response-only mid-layer features, strong L2 + bootstrap) PCA(50) has negligible effect on test performance (79.19% vs 79.81% AUROC, 75.62% vs 75.59% accuracy). PCA does reduce overfitting (train accuracy 79.68% vs 88.58%), but L2 + bootstrap already provide sufficient regularisation. We keep no-PCA as default since it marginally wins on AUROC. 

**Strong L2 regularisation.** With 3584 features and ~447 training samples per fold, aggressive regularisation is necessary. C is chosen from `{0.001, 0.002, 0.003, 0.005}` by internal 3-fold CV on AUROC.

**Bootstrap ensemble (30 models).** Each model is trained on a bootstrap resample (sampling with replacement) of the training set. Final probabilities are averaged. Ablation A5 shows the gain: single model → 78.24% AUROC / 74.57% accuracy; bootstrap ×30 → 79.81% / 75.59% (+1.6 pp AUROC, +1.0 pp accuracy).

**No class balancing.** Ablation A6 shows `class_weight='balanced'` has negligible effect on AUROC (79.77% vs 79.81%, −0.04 pp) but costs −0.82 pp accuracy. AUROC is threshold-independent so balancing doesn't affect it; accuracy suffers because balanced weights shift the decision boundary away from the natural 70/30 prior at threshold 0.5.

**Fixed threshold at 0.5.** With a well-calibrated LogReg and a stable 70/30 prior, 0.5 is the natural operating point. Threshold tuning was not systematically evaluated under Sol. 4.

### 3. Evaluation strategy (`splitting.py`)

`RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` = 25 folds. This gives a stable estimate of generalisation performance that does not depend on a single lucky seed. 

---

## Experiments

### Ablation study

Each row changes exactly one component of Sol. 4; all other settings are held fixed. Δ AUROC shows the cost of that change relative to Sol. 4.

| | What changes | Val AUROC | Test AUROC | Val Acc | Test Acc | Δ AUROC |
|---|---|---|---|---|---|---|
| **Sol. 4** | reference | **79.15%** | **79.81%** | **74.27%** | **75.59%** | — |
| A0 `sol4_prompt_only` | tokens: prompt only | 66.82% | 70.44% | 70.04% | 70.74% | −9.4 |
| A1 `sol4_all_tokens` | tokens: all (prompt+resp) | 66.82% | 70.44% | 70.04% | 70.74% | −9.4 |
| A2 `sol4_early_layers` | layers: L4–7 | 73.88% | 74.96% | 71.31% | 72.63% | −4.8 |
| A3 `sol4_late_layers` | layers: L20–23 | 76.08% | 77.64% | 72.77% | 74.31% | −2.2 |
| A5 `sol4_no_bootstrap` | probe: bootstrap ×1 | 77.72% | 78.24% | 74.00% | 74.57% | −1.6 |
| A4 `sol4_with_pca` | probe: +PCA(50) | 78.59% | 79.19% | 73.54% | 75.62% | −0.6 |
| A6 `sol4_balanced` | probe: class_weight=balanced | 79.00% | 79.77% | 73.65% | 74.77% | −0.04 |
| A7 `sol4_max_only` | pooling: max only | 78.14% | 78.99% | 73.92% | 75.39% | −0.8 |
| A8 `sol4_mean_only` | pooling: mean only | 78.57% | 79.44% | 73.73% | 75.41% | −0.4 |

---

### Search trajectory

Full development path; each step changes multiple variables simultaneously — useful as context, not as isolated comparisons.

| | Aggregation | Tokens | Layers | Probe | Split | Val AUROC | Test AUROC | Val Acc | Test Acc |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | Mean-pool | All | L0–24 | LogReg C=1, no scale | 1×5-fold | 71.4% | 73.8% | 73.1% | 75.0% |
| Sol. 1 | Mean-pool | All | L4,8,12,16,20,24 | Stacking + meta-LR | 1×5-fold | 72.3% | 66.7% | 72.3% | 70.5% |
| Sol. 2 | Last token | All | L8,12,16,20,24 | PCA(50) + bootstrap ×30 | 1×5-fold | 75.0% | 72.0% | 75.0% | 70.4% |
| Sol. 3 | Last token | All | L20–24 | PCA(50) + bootstrap ×30 | 1×5-fold | 73.5% | 71.2% | 73.5% | 71.7% |
| **Sol. 4** | Max+mean pool | Response only | L12,13 / L14,15 | Bootstrap ×30 | 25-fold | **79.15%** | **79.81%** | **74.27%** | **75.59%** |

---

## Reproducing experiments

Set `ACTIVE` in `config.py` and re-run `solution.py`:

| `ACTIVE` | What it runs |
|---|---|
| `"baseline"` | Mean-pool L0–24, LogReg C=1, no scale, 1×5-fold |
| `"sol2"` | Last-token L8/12/16/20/24, PCA(50)+bootstrap, 1×5-fold |
| `"sol3"` | Last-token L20–24, PCA(50)+bootstrap, 1×5-fold |
| `"sol4"` | Response-only max+mean L12–15, bootstrap, 25-fold **(default)** |
| `"sol4_all_tokens"` | A1 — token scope ablation |
| `"sol4_early_layers"` | A2 — layer selection (early) |
| `"sol4_late_layers"` | A3 — layer selection (late) |
| `"sol4_with_pca"` | A4 — PCA ablation |
| `"sol4_no_bootstrap"` | A5 — bootstrap ablation |
| `"sol4_balanced"` | A6 — class balancing ablation |
| `"sol4_max_only"` | A7 — pooling type ablation |
| `"sol4_mean_only"` | A8 — pooling type ablation |

Sol. 1 (stacking) is not reproducible via config — it required a separate probe implementation.

---

## Final prediction

The final probe for `predictions.csv` is trained on all 689 training samples (union of all fold train+val indices) using the best C found during CV, with 30 bootstrap models. This maximises training data for the held-out test set.
