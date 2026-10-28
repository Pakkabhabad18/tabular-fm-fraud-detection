# Evaluation questions and answers

Questions I expect at the mid and final evaluations, with the answer and where the
evidence is. Numbers marked *(seed 0)* are preliminary until all 5 seeds are in.

---

## The problem

**Q. What is the project in one sentence?**
Whether tabular foundation models, which predict from a table of labelled examples
without any training, can match a tuned XGBoost on card fraud when fraud is rarer
than 1%, and how much that depends on which rows they are shown.

**Q. Why is this not already answered?**
The benchmarks where TFMs beat gradient-boosted trees (OpenML-CC18, TALENT,
TabArena) either exclude datasets with a minority class below 5% or report accuracy
and ROC-AUC. The one imbalance benchmark that goes down to 0.17% (CLIMB) does not
include TFMs. So the fraud regime is exactly where neither side has measured.

**Q. What is "in-context learning" for tables?**
The model gets a set of labelled rows (the context) together with the rows to
predict, and outputs predictions in one forward pass. No weights are updated. It is
the tabular version of giving an LLM examples in the prompt. TabPFN was pretrained
on millions of synthetic tables so that this forward pass approximates Bayesian
prediction.

**Q. Why is context construction the core of the project?**
The weights are frozen, so the only thing I can change is which rows the model sees.
TabPFN v2 handles about 10K context rows. At 0.172% fraud, 10K random rows contain
about 17 frauds, so most of the fraud information is simply never shown to the model.

---

**Q. If the prediction depends on the context, how do you give the context, and isn't the answer just random?**
The model has no training step: `fit` only stores the context rows and labels, and
at prediction time each test row attends to all of them. So yes, a different context
gives different predictions. That dependence is what the project studies, and it is
controlled in four ways:
1. Every context is built from a seed by a fixed rule and its row indices are saved, so a run can be repeated exactly.
2. XGBoost is trained on the same saved rows, so both families see identical data.
3. Five seeds give five different contexts, so the reported ± std includes context noise. A separate test redraws only the context on a fixed split to measure that noise on its own.
4. The W1 policies replace a random choice with a deliberate one (balanced, kNN), and the ensemble policies average 5 contexts so no single draw decides the answer.

**Q. Isn't that the same as XGBoost giving a different model on different training data?**
Yes, it's the same idea. The difference is cost and size: XGBoost can use all 170K
training rows, while a TFM's context is limited to about 10K. Which 10K is chosen
therefore matters much more for a TFM.

---

## Data and splits

**Q. Why ULB, and how do you get different fraud rates?**
ULB is real card data with 0.172% fraud (492 of 284,807). I make higher rates by
dropping legitimate rows inside each split, keeping every fraud. That way the
features, frauds and preprocessing stay fixed and only the fraud rate changes.

**Q. Why are the levels "nested"?**
For each seed the legitimate rows are shuffled once and each fraud rate takes a
prefix of that order, so the 20% rows are a subset of the 5% rows, and so on. A drop
in PR-AUC from one level to the next is then caused by the fraud rate, not by a
different random sample of legitimate rows.

**Q. Why 5 seeds?**
Each test split has only ~98 frauds, so PR-AUC moves by about ±0.02 just from which
rows are drawn. Five repeats give a mean and a spread, and any difference smaller
than the spread is not treated as real.

**Q. Isn't a random stratified split unrealistic for fraud?**
Yes, fraud patterns drift over time, and a time-ordered split would be harder. I
followed the proposal's stratified protocol for the course and plan a time-ordered
check on ULB's `Time` column as a robustness test.

**Q. How do you know every model was tested on the same rows?**
The split indices are saved once and fingerprinted (`431e76c7a7e44ec0`). Every
notebook checks the fingerprint before running, and the paired tests assert that the
test rows are identical for the two models being compared.

---

## Metrics

**Q. Why not accuracy?**
Predicting "not fraud" for every transaction scores 99.83% accuracy and catches
nothing.

**Q. Why PR-AUC and not ROC-AUC?**
ROC-AUC uses the false-positive *rate*, which stays tiny when legitimate rows vastly
outnumber frauds, even if most alerts are false. In my XGBoost results ROC-AUC stays
at about 0.98 from 20% down to 0.172%, while PR-AUC falls from 0.97 to 0.87.
PR-AUC is built on precision, which is what an analyst feels: how many alerts are wasted.

**Q. What does a PR-AUC of 0.865 mean?**
Averaged over all thresholds, precision is about 0.87 at the points where each fraud
is caught. A model with no skill scores the fraud rate (0.0017 at 0.172%), so 0.865
is about 500 times better than chance.

**Q. Why do recall at top 1% and precision at the 0.1% budget look odd at 20%?**
At 20% the test set has 490 rows, so the top 1% is 5 rows. Recall can't exceed
5 / 98 = 0.051, and precision at a 0.1% budget is judged on a single row. These
metrics only mean something at 1% and below, so I only report them there.

**Q. How is the F1 threshold chosen?**
On validation, then applied unchanged to test. Choosing it on test would overstate F1.

**Q. How do you test whether a difference is real?**
Paired bootstrap per seed: both models are scored on the same resampled test rows
1,000 times, and I look at the 95% interval of the difference. Frauds and legitimate
rows are resampled separately so every resample has exactly 98 frauds.

---

## Models and fairness of the comparison

**Q. Which exact models do you use?**
TabPFN v2 (original checkpoint from the Nature paper, `tabpfn-v2-classifier.ckpt`)
and TabICLv2 (`tabicl-classifier-v2-20260212.ckpt`), with frozen weights. The
current `tabpfn` package defaults to v3.5, a different model, and its default v2
file was fine-tuned on real data, so I pin the original explicitly in `models.py`.

**Q. How strong is the XGBoost baseline?**
Three ways of handling imbalance (plain + tuned threshold, `scale_pos_weight`,
SMOTE), each tuned with 50 Optuna trials per fraud rate on validation PR-AUC with
early stopping. The test set is never used in tuning.

**Q. Why does SMOTE not help?**
Once the decision threshold is tuned, the trees already handle the imbalance, so all
three variants land within one standard deviation of each other. SMOTE also gives
worse probabilities (ECE about 0.02-0.03 vs 0.001 for plain at 1% and 0.5%), because
it trains on more frauds than really exist.

**Q. Isn't it unfair that XGBoost sees 170K rows and the TFM only 10K?**
That is why there is a matched-budget XGBoost: the same k rows the TFM saw. Comparing
TFM to matched XGBoost isolates the model family. Comparing TFM to full-data XGBoost
answers the practical question: is a TFM worth using at all on this data?

**Q. Why was XGBoost retuned?**
At first matched XGBoost scored exactly chance whenever its rows held only 1-2
frauds. I checked the cause on synthetic data. The search space started
`min_child_weight` at 1, which with ~0.1% frauds forces every leaf to hold ~1,000
rows, so no split could isolate the frauds. I widened the range to [0.001, 20] and
retuned both XGBoost arms. Full-data XGBoost moved by at most 0.009 (less than the
seed spread), so the fix only matters for tiny training sets.

**Q. Why use the largest k in the headline instead of the best k?**
Picking the best k after seeing test scores would flatter the TFMs. The largest k
that fits is a rule fixed in advance.

---

**Q. Which context policies did you compare, and why those?**
Uniform (baseline), stratified (keeps the fraud rate, removes draw luck), balanced
(all training frauds up to half the context), kNN and kNN with all frauds
(query-conditional retrieval, as in LoCalPFN), and ensembles of 5 uniform or 5
balanced contexts averaged in log-odds (the EasyEnsemble idea applied to contexts).
Each one targets a specific reason the uniform context fails below 1%: too few
frauds, frauds that aren't relevant to the row being scored, or too little of the
training data used.

**Q. Why is kNN done per cluster and not per test row?**
One context per test row would be ~57,000 model fits at 0.172%. I cluster the rows
to predict (about 1,000 per cluster) and retrieve the nearest training rows to each
cluster centre, in standardised features so Amount and Time don't dominate the
distance.

---

**Q. Are the predicted probabilities meaningful, or only the ranking?**
PR-AUC only uses the ranking. For probabilities I measure ECE and Brier score. Full-data
XGBoost is trained on the real fraud rate and should be close to calibrated. A TFM's
probabilities follow its context's fraud rate, so a balanced context (~3% fraud)
over-predicts at a 0.172% real rate. I correct this with Bayes' rule (prior
correction), which rescales the probabilities without changing their order, so PR-AUC
is unaffected.

**Q. What threshold would you use in practice?**
I compare four on the same test rows: 0.5 (library default), π₁ (McDowell et al.'s
rule, tested only down to 5%), the F1-best threshold from validation, and a fixed
alert budget of the top 0.1% of transactions. A fraud team usually works to a budget
(how many cases they can review per day), so the budget rule answers the practical
question: of the ~57 alerts a day, how many are real fraud?

**Q. Why does the prior correction not change PR-AUC?**
It is a monotone function of the probability: if p_a > p_b before, then p'_a > p'_b
after. PR-AUC depends only on the order of the scores.

---

## Results so far

**Q. What is the main result so far?**
Full-data XGBoost: 0.966 at 20% down to 0.865 ± 0.022 at 0.172%. TFMs with a uniform
context keep up at 20% and 5% (best 0.963 and 0.931) but fall about 0.10 behind at
0.172% (best 0.767) *(seed 0)*.

**Q. Why do the TFMs fall behind below 1%?**
A 10K uniform context at 0.172% contains only 19 of the 295 training frauds
*(seed 0)*. The model is shown very little of the class it has to find. This is the
problem class-aware contexts (W1) are designed to fix.

**Q. What is the most surprising observation?**
With a single fraud among 1,000 context rows, TabICLv2 reached PR-AUC 0.63
*(seed 0)*. Comparing a new row against the one fraud it has seen is enough for a
useful ranking. Whether XGBoost can match this with a properly tuned
`min_child_weight` is being tested in the current runs.

**Q. Which TFM is better?**
So far they are close in PR-AUC, with no consistent winner across fraud rates.
TabICLv2 is about 8× faster (0.61 vs 5.1 seconds per 1,000 predictions at k = 10K),
so it is the more practical one.

**Q. Can a TFM be used in a real fraud system?**
Not inline at card authorisation as is: 0.6-5 seconds per 1,000 transactions on a
GPU is far slower than tree inference. It could fit in an offline review queue, or
as a cold-start model for a new market with only a handful of labelled frauds,
where it needs no training. The final report will give a verdict once latency is
measured across dataset sizes (W4).

---

## Mistakes and limitations

**Q. Did anything go wrong?**
Two things, both caught and fixed before drawing conclusions:
1. Context sizes were first drawn independently, so a 2K context could hold fewer frauds than a 1K one. I made them nested.
2. XGBoost's search space made it unable to learn from 1-2 frauds. I widened it and retuned.

Both are in `docs/PROJECT_LOG.md`, with the old results kept as `results/archive_*`.

**Q. What are the main limitations?**
- One real dataset for the main sweep (IEEE-CIS comes later as a realism check).
- Stratified rather than time-ordered splits.
- XGBoost tuned on seed 0 and reused for seeds 1-4.
- Only ~98 test frauds per split, so small differences can't be resolved.

---

## What comes next

**Q. What will you show at the final evaluation that isn't here yet?**
- All 5 seeds for the TFMs.
- The W1 context-policy comparison (stratified, class-balanced, kNN, hard negatives, PCA, clustering, ensembles of disjoint contexts) with effect sizes.
- W2: calibration and the τ = π₁ threshold at 0.172%.
- W4: dataset size and context size against both score and latency.
- The IEEE-CIS realism check.
