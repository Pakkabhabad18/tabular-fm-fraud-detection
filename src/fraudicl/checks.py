"""Sanity checks: are the foundation-model scores real, or inflated by leakage or
memorisation?

A foundation model never trains on our data, so it can't overfit in the usual
sense (fitting noise in the training rows). But a high score could still be fake
in four ways, and each function here tests one of them:

1. context_leakage
   A test row slipped into the context. Then the model is shown the answer.
   Check: every saved context must lie inside its split's training rows and share
   no row with validation or test.

2. duplicate_overlap / rescore_without_duplicates
   ULB has 1,081 duplicated transactions (same Time, Amount and V1-V28). A test row
   with an exact copy in the training rows can be "recognised" instead of predicted,
   which helps a model that compares rows (a TFM attends to its context rows) even
   when no index overlaps. Check: count such rows, then recompute PR-AUC with them
   removed from the test set. If the scores barely move, duplicates aren't driving
   the result.

3. shuffled_label_control
   The model might "know" this dataset from pretraining. TabPFN v2 (the original
   checkpoint pinned in models.py) and TabICLv2 were both pretrained only on
   synthetic tables, so it shouldn't. Check: give the model the same context rows
   but with the labels shuffled. A model that uses the labels it is shown drops to
   chance (PR-AUC = fraud rate). A model that had memorised ULB would still score
   well.

4. matched_tests (in report.py terms: is a difference real?)
   A higher mean over 5 seeds can be luck. Check: paired bootstrap per seed, plus a
   pooled interval over all seeds, with Holm's correction because many cells are
   tested at once.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .metrics import pr_auc

LEVEL_ORDER = ["20", "5", "1", "0.5", "natural"]


# ---------------------------------------------------------------- 1. index leakage

def context_leakage(out_dir, splits) -> pd.DataFrame:
    """One row per saved context file: how many of its rows are outside the
    training rows, and how many are validation or test rows. All three counts
    must be 0.

    Saved contexts cover the uniform, stratified and balanced policies. kNN and
    ensemble contexts are built inside the run from split.train only and are not
    saved; tests/test_checks.py checks those builders directly.
    """
    by_key = {(s.seed, s.level): s for s in splits}
    folder = os.path.join(out_dir, "raw", "contexts")
    rows = []
    for f in sorted(os.listdir(folder)) if os.path.exists(folder) else []:
        if not f.endswith(".npy"):
            continue
        # file name: <policy>__<level>__k<k>__seed<seed>.npy
        policy, level, k, seed = f[:-4].split("__")
        s = by_key[int(seed[4:]), level]
        ctx = np.load(os.path.join(folder, f))
        rows.append({"policy": policy, "level": level, "k": int(k[1:]), "seed": int(seed[4:]),
                     "rows": len(ctx),
                     "outside_train": int(np.setdiff1d(ctx, s.train).size),
                     "in_validation": int(np.intersect1d(ctx, s.val).size),
                     "in_test": int(np.intersect1d(ctx, s.test).size)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. duplicated rows

def row_keys(X: np.ndarray) -> np.ndarray:
    """A 64-bit hash of each row's feature values. Two rows get the same key only
    if every feature is identical (a hash collision between different rows is
    about 1 in 10^19 per pair, so it can be ignored)."""
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


def duplicate_overlap(X, y, splits) -> pd.DataFrame:
    """Per split: how many test rows (and test frauds) have an exact copy among
    that split's training rows."""
    keys = row_keys(X)
    rows = []
    for s in splits:
        dup = np.isin(keys[s.test], keys[s.train])
        rows.append({"seed": s.seed, "level": s.level,
                     "test_rows": len(s.test), "test_rows_with_copy_in_train": int(dup.sum()),
                     "test_frauds": int(y[s.test].sum()),
                     "test_frauds_with_copy_in_train": int((dup & (y[s.test] == 1)).sum())})
    return pd.DataFrame(rows)


def rescore_without_duplicates(out_dir, X, y, splits,
                               families=("xgb", "tfm", "xgb_matched")) -> pd.DataFrame:
    """PR-AUC of every saved run, recomputed after dropping the test rows that have
    an exact copy in the training rows. Same scores, fewer test rows: nothing is
    refitted."""
    keys = row_keys(X)
    by_key = {(s.seed, str(s.level)): s for s in splits}
    out = []
    for family in families:
        folder = os.path.join(out_dir, "raw", family)
        if not os.path.exists(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".json"):
                continue
            meta = json.load(open(os.path.join(folder, f)))
            d = np.load(os.path.join(folder, f.replace(".json", ".npz")))
            s = by_key[int(meta["seed"]), str(meta["level"])]
            keep = ~np.isin(keys[d["test_idx"]], keys[s.train])
            yt = y[d["test_idx"]]
            out.append({"family": family, "model": meta["model"],
                        "policy": meta.get("policy", "all rows"),
                        "level": str(meta["level"]), "k": meta.get("k"), "seed": int(meta["seed"]),
                        "frauds_dropped": int((~keep & (yt == 1)).sum()),
                        "pr_auc": pr_auc(yt, d["test_scores"]),
                        "pr_auc_without_duplicates": pr_auc(yt[keep], d["test_scores"][keep])})
    df = pd.DataFrame(out)
    if len(df):
        df["change"] = df.pr_auc_without_duplicates - df.pr_auc
    return df


# ---------------------------------------------------------------- 3. shuffled labels

def shuffled_label_control(X, y, split, ctx, make, model_name, seed=0) -> pd.DataFrame:
    """Fit the model twice on the same context rows: once with the real labels,
    once with the labels randomly permuted (same number of frauds, attached to the
    wrong rows). Returns test PR-AUC for both, next to the no-skill level."""
    from .tfm import predict_in_batches

    rng = np.random.default_rng(seed)
    out = []
    for labels, y_ctx in (("real", y[ctx]), ("shuffled", rng.permutation(y[ctx]))):
        model = make(seed)
        model.fit(X[ctx], y_ctx)
        p = predict_in_batches(model, X[split.test])
        out.append({"model": model_name, "level": split.level, "k": len(ctx), "seed": split.seed,
                    "context_labels": labels, "context_frauds": int(y_ctx.sum()),
                    "pr_auc": pr_auc(y[split.test], p),
                    "no_skill_pr_auc": float(y[split.test].mean())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 4. is a difference real?

def holm(pvalues) -> np.ndarray:
    """Holm's step-down correction for testing many cells at once. Returns adjusted
    p-values; a cell is significant at 5% if its adjusted p-value is below 0.05."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])   # never smaller than the previous one
        adj[i] = min(1.0, running)
    return adj


class _SortedAP:
    """Average precision (the same number as sklearn's average_precision_score) for
    many bootstrap resamples of one test set, without re-sorting each time.

    A bootstrap resample with replacement is the same as giving each test row a
    weight = how many times it was drawn. The order of the scores never changes,
    so we sort once and each resample is just a weighted running sum: about 50x
    faster than calling sklearn in a loop on 57K rows.
    """

    def __init__(self, y, s):
        order = np.argsort(-np.asarray(s), kind="stable")
        self.order = order
        self.y = np.asarray(y)[order].astype(float)
        s_sorted = np.asarray(s)[order]
        # tied scores form one threshold; keep the last row of each run of ties
        self.cut = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]

    def __call__(self, w_sorted):
        tp = np.cumsum(w_sorted * self.y)[self.cut]
        flagged = np.cumsum(w_sorted)[self.cut]
        # thresholds above every drawn row have no weight yet; they add nothing
        # to the area (recall doesn't move there), so their precision is set to 0
        precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
        recall = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def pooled_paired_bootstrap(pairs, n_boot=1000, seed=0) -> dict:
    """Mean PR-AUC difference (A - B) over several seeds, with a 95% interval.

    pairs: list of (y_test, scores_a, scores_b), one per seed, both models scored on
    the same test rows. In each bootstrap round every seed's test set is resampled
    (frauds and legitimate rows separately, as in metrics.paired_bootstrap) and the
    seed-level differences are averaged. This uses the evidence of all seeds at
    once, which a per-seed interval can't do.

    What it does NOT cover: a different random split or context draw. That source
    of noise is what the across-seed std shows, and it's why a claim needs fresh
    seeds (a confirmatory run), not just this interval.
    """
    rng = np.random.default_rng(seed)
    observed = np.mean([pr_auc(yt, a) - pr_auc(yt, b) for yt, a, b in pairs])
    boots = np.zeros(n_boot)
    for yt, a, b in pairs:
        ap_a, ap_b = _SortedAP(yt, a), _SortedAP(yt, b)
        pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
        for i in range(n_boot):
            # stratified resample as row weights: frauds and legitimate rows are
            # drawn separately, so every resample keeps the same number of frauds
            w = np.zeros(len(yt))
            w[pos] = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos))
            w[neg] = np.bincount(rng.integers(0, len(neg), len(neg)), minlength=len(neg))
            # same weights for both models: that is what makes it paired
            boots[i] += ap_a(w[ap_a.order]) - ap_b(w[ap_b.order])
    boots /= len(pairs)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"mean_diff": float(observed), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p)}


def _test_scores(out_dir, family, name):
    path = os.path.join(out_dir, "raw", family, f"{name}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["test_idx"], d["test_scores"]


def matched_tests(out_dir, y, runs: pd.DataFrame, models=("TabICLv2", "TabPFN v2"),
                  policy="uniform", n_boot=1000) -> pd.DataFrame:
    """TFM vs XGBoost trained on exactly the same k rows, for every fraud rate and
    k. Per cell: mean difference over seeds, how many seeds favour the TFM, the
    pooled 95% interval and p-value, and the Holm-adjusted p-value over all cells
    in the table."""
    rows = []
    sub = runs[(runs.family == "tfm") & (runs.policy == policy)]
    for (model, level, k), g in sub.groupby(["model", "level", "k"], observed=True):
        if model not in models:
            continue
        tag = model.lower().replace(" ", "")
        pairs, frauds = [], []
        for seed in sorted(g.seed.unique()):
            a = _test_scores(out_dir, "tfm", f"{tag}_{policy}__{level}__k{int(k)}__seed{seed}")
            b = _test_scores(out_dir, "xgb_matched",
                             f"xgb_plain_matched_{policy}__{level}__k{int(k)}__seed{seed}")
            if a is None or b is None:
                continue
            assert np.array_equal(a[0], b[0]), "test rows differ between models"
            pairs.append((y[a[0]], a[1], b[1]))
            frauds.append(float(g[g.seed == seed].context_frauds.iloc[0]))
        if not pairs:
            continue
        per_seed = [pr_auc(yt, a) - pr_auc(yt, b) for yt, a, b in pairs]
        pooled = pooled_paired_bootstrap(pairs, n_boot=n_boot, seed=int(k))
        rows.append({"model": model, "level": str(level), "k": int(k), "n_seeds": len(pairs),
                     "context_frauds_mean": float(np.mean(frauds)),
                     "seeds_tfm_ahead": int(sum(d > 0 for d in per_seed)), **pooled})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df.p_value)
        df["level"] = pd.Categorical(df.level, LEVEL_ORDER, ordered=True)
        df = df.sort_values(["model", "level", "k"]).reset_index(drop=True)
    return df
