# In-Context Learning Below One Percent

**Tabular foundation models vs gradient-boosted trees for card fraud detection**

Prakash Bhabad · IIIT Hyderabad · 2026

> **Status:** proposal stage. The full proposal is in [`docs/proposal.pdf`](docs/proposal.pdf). Code, experiments and results will be added to this repository as the work progresses.

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
