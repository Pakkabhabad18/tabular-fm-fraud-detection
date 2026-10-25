# Project log

Running record of the project: what I did at each step, why, what came out, what it
means, and what I would change or what could have gone differently. Newest entries
are at the bottom. Numbers are test-set PR-AUC unless stated otherwise.

Short glossary used throughout:

- **TFM**: tabular foundation model (TabPFN v2, TabICLv2). Pretrained once, never trained on our data.
- **Context**: the labelled rows handed to a TFM at prediction time. It plays the role a training set plays for XGBoost.
- **k**: number of rows in the context (1K, 2K, 5K or 10K).
- **π₁ (fraud rate)**: share of fraud in train, validation and test. Natural ULB rate is 492 / 284,807 = 0.172%.
- **Matched-budget XGBoost**: XGBoost trained on exactly the k context rows the TFM saw.

---

## Aug 2026: proposal

**What.** Proposed comparing TabPFN v2 and TabICLv2 (inference only) against tuned
XGBoost for card fraud, measured in PR-AUC, as the fraud rate falls from 20% to the
natural 0.172% of the ULB dataset. Core idea: since the TFM weights are frozen, the
only lever is which rows go into the context ("context construction").

**Why.** Published results where TFMs beat gradient-boosted trees come from
benchmarks that either drop heavily imbalanced datasets or score with accuracy and
ROC-AUC. Both stay high under imbalance even when most alerts are false. Fraud is
exactly the case those benchmarks skip.

**Key design choices and the reason for each.**

| Choice | Reason |
| --- | --- |
| PR-AUC as the headline metric | Accuracy is 99.83% for a model that never flags fraud. ROC-AUC barely moves with the fraud rate. PR-AUC drops when false alarms pile up, which is what a fraud team feels. |
| Prevalence sweep made inside ULB (downsampling negatives) | Keeps features, frauds and preprocessing fixed, so the fraud rate is the only thing that changes. Using different datasets for different rates would mix in dataset differences. |
| Matched-budget XGBoost | A TFM sees only k rows, XGBoost sees all ~170K. Training XGBoost on the same k rows separates "which model family" from "how much data". Earlier work (García et al.) gave the TFMs balanced contexts and the trees raw data, so the two effects were mixed. |
| 70% modern / 30% classical | Course requirement. The modern side is the context-construction study; the classical side is a properly tuned XGBoost. |

---

## 4 Oct: proposal accepted, related work checked

Accepted as a solo project. Before starting I checked what had appeared since the
proposal:

- **Tanna et al. (May 2026)** test 7 context-sampling strategies for TFMs on credit
  default data at 8-22% default rates and find the sampling strategy matters more
  than the model choice. Close to my W1, but at much higher minority rates and with
  ROC-AUC.
- **McDowell et al. (May 2026)** show TabPFN is biased toward the majority class and
  that thresholding at τ = π₁ fixes it, tested down to 5%. My RQ3 extends this.
- **Prior Labs' own fraud notebook** reports TabPFN 3.5 at 0.901 AUPRC vs 0.838 for
  XGBoost on ULB, using a random 80/20 split and the hosted API.

**What it means for my project.** The gap I fill is below 1% fraud, with PR-AUC,
matched budgets and open models. These three works go in the related-work section
of the report.

---

## 17 Oct: setup and smoke test (task T1)

**What.** Set up the repo, pinned library versions, and wrote `models.py` so every
notebook builds the models the same way. Ran a smoke test on a Colab T4.

**Problem I found while setting up.** The current `tabpfn` package (9.2.0) loads
**TabPFN v3.5** by default, a different and gated model. Its default "v2" checkpoint
is also one that was fine-tuned on real datasets. My proposal describes TabPFN v2 as
pretrained on synthetic data only (Hollmann et al., Nature 2025), so I pinned the
original checkpoint `tabpfn-v2-classifier.ckpt`. TabICL is pinned to
`tabicl-classifier-v2-20260212.ckpt`.

**What I got** (`results/smoke_test.json`, 10K rows, 4.9% fraud):

| Model | PR-AUC | Seconds per 1K predictions at k = 10K | Peak GPU memory |
| --- | --- | --- | --- |
| TabPFN v2 | 0.938 | 5.1 | 4.9 GB |
| TabICLv2 | 0.951 | 0.61 | 9.5 GB |
| XGBoost (defaults) | 0.940 | ~0 | none |

**What it means.** The scores only show the pipeline works (small sample, inflated
fraud rate). The timing matters: TabICLv2 is about 8× faster than TabPFN v2, and
both are thousands of times slower than tree inference. That is a deployment point
for the report: a TFM at 5 s per 1,000 transactions cannot sit inline at card
authorisation without batching.

---

## 19 Oct: data and splits (task T2)

**What.** Stratified 60/20/20 train/validation/test split of all 284,807 rows, for
5 seeds. For each fraud rate I keep every fraud and drop legitimate rows until the
rate is reached. Split indices are saved and fingerprinted (`431e76c7a7e44ec0`), and
every later notebook checks the fingerprint before running.

**Design choice: nested levels.** Per seed the legitimate rows are shuffled once,
and each fraud rate takes a prefix of that order. So the rows at 20% are a subset of
the rows at 5%, and so on. A change in PR-AUC between two levels then comes from the
fraud rate, not from a different random sample.

**What I got** (`results/ulb_split_counts.csv`, same counts for every seed):

| Fraud rate | Train rows (frauds) | Test rows (frauds) |
| --- | --- | --- |
| 20% | 1,475 (295) | 490 (98) |
| 5% | 5,900 (295) | 1,960 (98) |
| 1% | 29,500 (295) | 9,800 (98) |
| 0.5% | 59,000 (295) | 19,600 (98) |
| 0.172% | 170,883 (295) | 56,962 (98) |

**What it means.**
- Only ~98 test frauds per split, so PR-AUC is noisy. That is why there are 5 seeds and paired bootstrap tests.
- At 20% the training pool is 1,475 rows, so context sizes 2K and up are impossible there. The grid is ragged by construction, not by mistake.

**What could differ.** I used stratified random splits as in the proposal. Real fraud
data drifts over time, so a time-ordered split (train on early transactions, test on
later ones) would be more realistic and probably lower every score. I plan this as a
robustness check after the course deadline.

---

## 19 Oct: scoring code (task T3)

**What.** One function (`metrics.all_metrics`) computes every metric for every
model, so the two families can't end up scored differently. 21 unit tests on cases
with known answers (a perfect model scores 1, a random one scores the fraud rate, and so on).

**Choices and reasons.**
- **F1 threshold picked on validation, applied to test.** Picking it on test would overstate F1.
- **ECE with equal-count bins.** At 0.17% fraud almost every predicted probability is near 0, so equal-width bins would put nearly all rows in the first bin and hide the errors.
- **Paired bootstrap resamples frauds and legitimate rows separately.** This keeps the fraud count fixed in every resample. PR-AUC depends on the fraud rate, so a plain bootstrap would add noise unrelated to the models.

---

## 19-20 Oct: tuned XGBoost (task T4)

**What.** Three variants (plain with a tuned threshold, `scale_pos_weight`, SMOTE on
training rows only), each tuned with 50 Optuna trials per fraud rate on seed 0
(validation PR-AUC, early stopping), then refit on all 5 seeds.

**What I got** (`results/xgb_summary.csv`, plain variant, mean ± std over 5 seeds):

| Fraud rate | PR-AUC | ROC-AUC |
| --- | --- | --- |
| 20% | 0.966 ± 0.009 | 0.983 |
| 5% | 0.930 ± 0.014 | 0.981 |
| 1% | 0.894 ± 0.016 | 0.985 |
| 0.5% | 0.881 ± 0.020 | 0.987 |
| 0.172% | **0.865 ± 0.022** | 0.985 |

**What it means.**
1. **ROC-AUC hides the difficulty.** It stays at ~0.98 across a 116× drop in fraud rate, while PR-AUC falls from 0.97 to 0.87. This is the proposal's argument, shown on real data.
2. **XGBoost is strong.** At 0.172% a random model scores 0.0017 and XGBoost 0.865, about 500× better.
3. **Reweighting and SMOTE add nothing to PR-AUC.** All three variants are within one std at every fraud rate. Once the threshold is tuned, the trees handle imbalance on their own. The tuned `scale_pos_weight` jumps between 1.5 and 14 across fraud rates, a sign it barely matters.
4. **SMOTE hurts calibration.** At 1% and 0.5%, SMOTE's ECE is about 0.02-0.03 against about 0.001 for plain. It trains on many more frauds than really exist, so its probabilities are too high. Ranking is unaffected, so PR-AUC doesn't show it.
5. **Alert metrics are meaningless at high fraud rates.** At 20%, the top 1% of 490 test rows is 5 rows, so recall can't exceed 5/98 = 0.051. I report these metrics only at 1% and below.

**What could differ.** Tuning on each seed separately would cost 5× more. Because the
variants and settings barely change PR-AUC, I don't expect it to matter, but it is a
stated simplification.

---

## 20 Oct: foundation models with a uniform context, seed 0 (task T5)

**What.** Each TFM gets k training rows drawn at random ("uniform"), for every k
that fits. XGBoost is also trained on exactly those k rows (task T6).

### Correction 1: contexts must be nested across k

In the first run each k was drawn independently. At 0.5% fraud the 2K context ended
up with fewer frauds (7) than the 1K context (8), so the curve over k jumped around
from sampling alone. I changed it so each seed shuffles the training pool once and k
takes the first k rows, the same logic as the nested fraud rates. The first run is
kept in `results/archive_t5_seed0_independent_contexts.csv`.

**What I got** (seed 0, nested contexts, `results/archive_t5_seed0_nested_xgb_search_v1.csv`):

| Fraud rate | Frauds in context at k = 1K / 2K / 5K / 10K | Best TFM | Full-data XGBoost |
| --- | --- | --- | --- |
| 20% | 192 (1K only) | 0.963 | 0.966 |
| 5% | 46 / 91 / 244 | 0.931 | 0.930 |
| 1% | 9 / 20 / 46 / 104 | 0.855 | 0.894 |
| 0.5% | 2 / 5 / 23 / 49 | 0.794 | 0.881 |
| 0.172% | 1 / 2 / 9 / 19 | 0.767 | 0.865 |

**What it means (one seed, so preliminary).**
1. **At 20% and 5% the TFMs keep up** with full-data XGBoost while seeing only k rows.
2. **Below 1% a uniform context falls behind.** At 0.172%, 10K random rows contain only 19 of the 295 training frauds. The gap to XGBoost (~0.10) is the cost of that. A class-balanced context could include all 295 frauds within the same 10K, which is what W1 tests.
3. **TFMs learn from almost nothing.** With **one** fraud among 1,000 context rows, TabICLv2 still reached PR-AUC 0.63 (chance is 0.0017).

### Correction 2: the XGBoost search space was unfair at tiny fraud counts

Matched XGBoost scored exactly the fraud rate (no skill) whenever its k rows held
only 1-2 frauds. Before calling this a finding, I tested the cause on synthetic data
(2,000 rows, frauds shifted in feature space):

| Frauds in training rows | `min_child_weight` = 1 | `min_child_weight` = 0.1 |
| --- | --- | --- |
| 1 | 0.002 | 0.979 |
| 2 | 0.002 | 0.343 |
| 20 | 0.972 | 0.954 |

**Cause.** `min_child_weight` is the minimum total hessian a tree leaf may hold. At
the start of boosting each row's hessian is p(1-p) ≈ 0.001 when frauds are ~0.1% of
rows, so a floor of 1 forces every leaf to hold ~1,000 rows. With 1-2 frauds no split
can separate them and the model predicts a constant. My search started at 1
(XGBoost's default), so it could never try smaller values.

**Fix.** Search range lowered to [0.001, 20] for both full-data and matched XGBoost,
both retuned. Old results kept as `results/archive_*search_v1*`.

**Check that the fix didn't change the main baseline.** Full-data XGBoost moved by at
most 0.009 in any cell, smaller than the seed-to-seed spread every time (natural:
0.864 → 0.865). With 295 training frauds the floor never mattered. It only matters
when the training rows hold a handful of frauds, which is exactly the matched-budget
setting.

**Why this matters for the conclusions.** Without the fix, "TFMs win when fraud
examples are scarce" would have rested on a baseline that was handicapped by my own
search space. If the gap survives the retuned matched XGBoost, the finding is real.

---

## 25 Oct: results code (task T7)

**What.** `report.py` and notebook 04 collect every saved run and produce the
headline table, the per-k table, paired bootstrap tests and three figures.

**Choice: which k represents a TFM in the headline.** The largest k that fits at each
fraud rate (1K at 20%, 5K at 5%, 10K below). This rule is fixed in advance. Picking
the best k per fraud rate after seeing test scores would flatter the TFMs.

**Status.** Full-data XGBoost has all 5 seeds. TFMs have seed 0. Next: seeds 1-4 of
notebook 03 (with matched XGBoost on the new search space), then notebook 04, then
the first W1 comparison (stratified, class-balanced, kNN, ensemble of contexts).

---

## 25 Oct: context policies (task T8, workstream W1)

### How the context is given, and why the answer depends on it

A TFM has no training step. `model.fit(X[ctx], y[ctx])` only stores the context rows
and their labels. At `predict_proba(X_test)` each test row is passed through the
network together with all context rows, and attention lets it compare itself with
every labelled example. So the context is the model's entire knowledge of the task:
**change the context and the predictions change.** This is not a flaw to hide; it is
the variable the project studies.

How I control it:

1. **Reproducible contexts.** Every context is built by a fixed rule from a seed, and
   its row indices are saved to Drive. Rerunning gives exactly the same rows, and the
   code refuses to reuse a saved context that the current rule would not produce.
2. **Same rows for the classical arm.** Matched XGBoost is trained on the saved
   context rows, so both families see identical data.
3. **Noise measured, not ignored.** Each split seed also draws a different context,
   so the ± std over 5 seeds includes context noise. A separate run holds the split
   fixed and redraws only the context 5 times, which measures context noise on its
   own.
4. **Choosing the context deliberately is the experiment.** W1 compares rules for
   choosing it. The ensemble policies average 5 different contexts, which both uses
   more of the training data and reduces the dependence on any single draw.

### Policies implemented

| Policy | Rule | Why test it |
| --- | --- | --- |
| uniform | k random training rows | baseline from T5 |
| stratified | fraud rate of the pool kept exactly, at least 1 fraud | removes the luck of a uniform draw (e.g. 0 frauds at k = 1K) |
| balanced | all training frauds up to k/2, rest random legitimate | below 1% fraud this shows the model all 295 frauds instead of ~19 |
| knn | per cluster of rows to predict, the k nearest training rows (standardised features) | the proposal's query-conditional retrieval |
| knn_balanced | all frauds up to k/2 plus the nearest legitimate rows | plain kNN can return no fraud at all near legitimate clusters |
| ensemble_uniform | 5 non-overlapping uniform contexts, log-odds averaged | uses 5× more training rows than one context holds |
| ensemble_balanced | 5 contexts sharing all frauds, each with its own legitimate rows | EasyEnsemble from classical imbalanced learning, applied to contexts |

All single-context policies are nested across k, like the uniform one.

**Choices and reasons.**
- **kNN per cluster, not per row.** A separate context per test row would mean ~57,000 model fits at 0.172%. Rows to predict are clustered (about 1,000 per cluster) and each cluster gets the training rows nearest its centre. Validation and test are clustered separately.
- **Standardised distances.** Amount and Time have far larger ranges than V1-V28 and would decide the neighbours on their own otherwise.
- **Ensembles only at k = 1K and 10K.** Each costs 5 single runs.
- **TabICLv2 for the full W1 sweep.** It is about 8× faster than TabPFN v2 and scored about the same in T5. TabPFN v2 runs the stratified and balanced policies on seed 0 if GPU time allows.
- **Effect sizes are paired by seed.** A policy is compared with uniform on the same split and the same test rows, which removes the noise both share.

**Expected result, to be checked.** Balanced and ensemble policies should close part
of the ~0.10 gap at 0.172% because they show the model many more frauds. kNN alone
may do worse there, because the nearest rows to most test clusters are legitimate.

---

## Open questions I am tracking

- Does the few-shot advantage hold across all 5 seeds once matched XGBoost uses the retuned search space?
- How much of the 0.10 gap at 0.172% does a class-balanced context close?
- Is the TFM gain from more context rows, or from more frauds in the context? (Planned follow-up: hold the number of frauds fixed and vary their share.)
- How large is context-only noise compared with split-to-split noise? (Measured in notebook 05, cell 6.)
- Do TFM probabilities stay calibrated at 0.172%, and does τ = π₁ still work as a threshold (RQ3)?
