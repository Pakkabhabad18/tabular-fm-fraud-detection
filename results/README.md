# results/

Small summary tables and figures. Raw per-run scores (`.npz`) and per-run metrics
(`.json`) stay on Google Drive under `results/raw/`; everything here is rebuilt from
them by notebook 04 or copied from the notebook printouts.

## Setup checks

| File | Made by | What it holds |
| --- | --- | --- |
| `smoke_test.json` | notebook 00 | first end-to-end run, timings on a T4 |
| `ulb_split_counts.csv` | notebook 01 | rows and frauds per split, seed and fraud rate |
| `ulb_splits_fingerprint.txt` | notebook 01 | hash of the split indices; every notebook checks it |

## Main results

| File | Made by | What it holds |
| --- | --- | --- |
| `xgb_summary.csv`, `xgb_best_params.jsonl` | notebook 02 | full-data XGBoost, 3 variants, mean ± std over 5 seeds; tuned settings |
| `t5_t6_uniform_by_k_5seeds.csv` | notebook 03 | TFMs and same-rows XGBoost for every fraud rate and k, uniform context |
| `headline_table.csv` | notebook 04 | each model at the largest k that fits (figure 1): PR-AUC, ROC-AUC, F1 at the validation threshold, recall at 0.1% false-positive rate |
| `by_k_table.csv` | notebook 04 | every model, fraud rate and k (figure 2), PR-AUC and F1 |
| `paired_tests.csv` | notebook 04 | each model vs full-data XGBoost, paired bootstrap per seed |
| `runs_all.csv` | notebook 04 | one row per saved run, all metrics |

## Test-mix control

| File | Made by | What it holds |
| --- | --- | --- |
| `rescore_by_test_mix_runs.csv` | notebook 07 | natural-rate models scored again on the 20%, 5%, 1% and 0.5% test sets, per seed |
| `rescore_by_test_mix_summary.csv` | notebook 07 | mean over seeds, next to the PR-AUC of the model built at each rate (figure 6) |

The per-run file was copied from the notebook printout (6 decimals); the summary and
figure 6 were rebuilt from it with `report.rescore_summary` and
`report.plot_rescore_by_test_mix` and match the Colab output to within 3e-7.

## Context policies (W1)

| File | Made by | What it holds |
| --- | --- | --- |
| `policy_table_tabiclv2.csv` | notebooks 04, 09 | all ten policies vs uniform, paired by seed (figure 4); the 09 printout adds cluster, pca and hard_negative |
| `context_variance.csv` | notebook 04 | 5 random contexts on one fixed split: noise from the context alone |
| `policy_same_rows_xgb.csv` | notebook 09 | TabICLv2 vs XGBoost trained on the same context rows, balanced and the three new policies |
| `knn_diagnosis_summary.csv` | notebook 09 | why kNN contexts fail: groups without fraud, ranking inside groups vs pooled, spread of score scales |

## Scale and speed (W4)

| File | Made by | What it holds |
| --- | --- | --- |
| `scale_natural.csv` | notebook 10 | PR-AUC and F1 at 0.172% against the number of rows each model learns from: XGBoost on 1K to all 170,883 rows, TabICLv2 contexts up to 50K, TabPFN v2 up to 10K (figure 7) |
| `latency.csv` | notebook 10 | time to score 1, 100 and 1,000 transactions, per model |

## Calibration and thresholds (W2)

| File | Made by | What it holds |
| --- | --- | --- |
| `w2_calibration_summary.csv`, `w2_calibration_runs.csv` | notebook 04 | ECE and Brier before and after prior correction (figure 5) |
| `w2_threshold_summary.csv`, `w2_thresholds_runs.csv` | notebook 04 | alerts, frauds caught, precision, recall, F1 for each threshold rule |

## Sanity checks

| File | Made by | What it holds |
| --- | --- | --- |
| `check_duplicates.csv` | notebook 06 | test rows and frauds with an exact copy in training |
| `check_shuffled_labels.csv` | notebook 06 | real vs shuffled context labels (memorisation test) |
| `check_tfm_vs_matched_xgb.csv` | notebook 06 | TFM vs same-rows XGBoost, pooled over seeds, Holm-corrected |

## Figures

| File | Shows |
| --- | --- |
| `figures/fig1_headline.png` | PR-AUC against fraud rate, all models |
| `figures/fig2_by_k.png` | PR-AUC against context size k, one panel per fraud rate |
| `figures/fig3_context_frauds.png` | score against the number of frauds in the context, at 0.172% |
| `figures/fig4_policies.png` | the context policies against k, at 0.172% and 1% |
| `figures/fig5_reliability.png` | calibration at 0.172% before and after prior correction |
| `figures/fig6_test_mix.png` | same natural-rate models on every test mix: how much of the PR-AUC drop is the test set itself |
| `figures/fig7_scale.png` | PR-AUC at 0.172% against rows seen: XGBoost on N rows vs foundation-model contexts of N rows |

## Superseded (kept for the record)

`archive_*` files are runs replaced after a fix: the first seed-0 runs with
independent (not nested) contexts, and XGBoost tuned with the old `min_child_weight`
range. `docs/PROJECT_LOG.md` explains both.
