# In-Context Learning Below One Percent

**Tabular foundation models vs gradient-boosted trees for card fraud detection**

Prakash Bhabad · IIIT Hyderabad · 2026

> **Status (25 Oct 2026):** classical arm done (5 seeds); foundation-model sweep in progress (seed 0 done). Proposal: [`docs/proposal.pdf`](docs/proposal.pdf). Step-by-step record: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md). Mid-evaluation: 2–4 Nov 2026. Final submission: 30 Nov 2026.

---

## The question

Tabular foundation models (TabPFN v2, TabICLv2) are pretrained once and then make predictions from a labelled table passed in as context, with no training on the new dataset. Published results show them matching or beating tuned gradient-boosted trees, but those benchmarks either exclude heavily imbalanced data or score on accuracy and ROC-AUC, which stay high even when a model catches very little of the minority class.

Card fraud sits exactly in that gap: in the ULB dataset only 0.17% of transactions are fraud. This project asks:

1. **Does the in-context learning advantage survive as the fraud rate drops from 20% to 0.17%, measured in PR-AUC?**
2. **Does the way the context is built matter more than the choice of model family?**
3. **Does the known prevalence-threshold correction, tested only down to 5%, still hold two orders of magnitude lower?**

## Why context construction is the core problem

The foundation models are used with frozen weights, so the only thing that can be changed is which rows go into the context. TabPFN v2 takes roughly 10K context rows; at a 0.17% fraud rate a random 10K-row sample holds about 17 frauds. The project compares several ways of building that context (uniform, stratified, class-balanced, nearest-neighbour retrieval, hard-negative mining, PCA- and clustering-based selection) across context sizes of 1K to 10K rows.

## Setup

| | Classical arm | Modern arm |
|---|---|---|
| Model | XGBoost, tuned with a 50-trial Optuna search | TabPFN v2 and TabICLv2, inference only |
| Imbalance handling | `scale_pos_weight`, SMOTE, threshold tuning | Context construction policy, ensembling over disjoint contexts |
| Fairness control | Also trained on the same k-row subsets the foundation models see | — |

Both arms share the same splits and the same scoring code. Training XGBoost on the same row budget separates the effect of the model family from the effect of how much data each model sees.

**Data:** ULB Credit Card Fraud (284,807 transactions, 0.17% fraud), with the fraud rate varied by downsampling negatives to 20%, 5%, 1%, 0.5% and 0.17%. IEEE-CIS Fraud Detection (590K transactions, ~3.5% fraud, 433 features) is used as a realism check.

## Evaluation

Accuracy is not used: predicting "no fraud" for every transaction already scores 99.83%.

- **Primary:** PR-AUC, with ROC-AUC alongside for comparison with published work, and recall when flagging only the top 1% of transactions
- **Operational:** precision at a 0.1% alert budget, recall at 0.1% false-positive rate
- **Calibration:** expected calibration error and reliability diagrams
- **Cost:** per-record inference latency and memory, against tree inference
- **Statistics:** five stratified repeats with paired bootstrap tests, since the dataset has only 492 frauds

A negative result with a supported explanation, for example that the advantage disappears below 1% and why, counts as a result.

## Planned outputs

- PR-AUC against fraud rate for both model families, showing where they change order
- A ranked comparison of context-construction policies with effect sizes
- A verdict on where a tabular foundation model can sit in a fraud system: inline at authorisation, or in the offline review queue

## Results so far

Full-data XGBoost (plain variant, mean ± std over 5 seeds) against the best foundation model with a uniform context (seed 0 only, preliminary):

| Fraud rate | XGBoost PR-AUC | Best TFM PR-AUC (seed 0) | Frauds in a 10K uniform context |
| --- | --- | --- | --- |
| 20% | 0.966 ± 0.009 | 0.963 | n/a (pool is 1,475 rows) |
| 5% | 0.930 ± 0.014 | 0.931 | n/a (k ≤ 5K) |
| 1% | 0.894 ± 0.016 | 0.855 | 104 |
| 0.5% | 0.881 ± 0.020 | 0.794 | 49 |
| 0.172% | 0.865 ± 0.022 | 0.767 | 19 |

- ROC-AUC stays near 0.98 at every fraud rate while PR-AUC falls, which is why PR-AUC is the headline metric.
- `scale_pos_weight` and SMOTE do not beat plain XGBoost with a tuned threshold, and SMOTE gives worse-calibrated probabilities.
- Below 1%, a uniform context shows the foundation models only a small share of the training frauds. Class-aware contexts (next step) target this gap.

Details, reasons and corrections: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md). Expected evaluation questions: [`docs/EVAL_QA.md`](docs/EVAL_QA.md).

## Repository layout

```
docs/          proposal, project log, evaluation Q&A
src/fraudicl/  project code
  config.py      seeds, split fractions, fraud rates, context sizes
  data.py        loading ULB, building and fingerprinting the splits
  metrics.py     every metric and the paired bootstrap
  models.py      model factories with pinned checkpoints
  xgb.py         XGBoost variants and Optuna tuning
  tfm.py         context policies for the foundation models (uniform, stratified,
                 balanced, kNN, ensembles)
  experiments.py resumable experiment loops (results saved per run)
  report.py      tables, figures and significance tests
  calibration.py W2: prior correction, ECE/Brier, threshold rules, reliability plot
  checks.py      sanity checks: context leakage, duplicated rows, shuffled labels,
                 TFM vs same-rows XGBoost significance (Holm-corrected)
tests/         unit tests (python -m pytest tests -q)
notebooks/     00 smoke test, 01 splits, 02 XGBoost, 03 foundation models (uniform context),
               04 results, 05 context policies (W1), 06 sanity checks
results/       summary tables and figures (raw per-run outputs stay on Google Drive);
               archive_* files are superseded runs kept for the record
```

Data is not stored in the repo. ULB is downloaded with `kagglehub` inside each notebook.

## Running

Run the notebooks in order on Colab (File → Open notebook → GitHub). 00, 02, 03, 05 and 06 need a T4 GPU; 01 and 04 run on CPU. Every experiment notebook saves each finished run to Google Drive and skips it on a rerun, so a disconnected session can simply be run again.

Model versions are pinned on purpose: TabPFN v2 uses the original checkpoint from Hollmann et al. (2025), not the package's current default (v3.5), and TabICL uses the TabICLv2 checkpoint `tabicl-classifier-v2-20260212.ckpt`.
