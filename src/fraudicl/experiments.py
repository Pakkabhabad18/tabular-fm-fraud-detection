"""Experiment loops that save as they go, so a Colab disconnect costs one model, not
the whole sweep. Rerunning a loop skips everything already saved.

Saved per run (on Drive, not in git):
  raw/<family>/<name>.npz    validation and test scores plus row indices, kept so the
                             paired bootstrap and later analyses can reuse them
  raw/<family>/<name>.json   the metrics row for that run
The small summary table built from the .json files is what gets committed.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .metrics import all_metrics, best_f1_threshold
from .xgb import fit_xgb, tune


def _save_run(out_dir, family, name, row, val_idx, val_s, test_idx, test_s):
    folder = os.path.join(out_dir, "raw", family)
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(
        os.path.join(folder, f"{name}.npz"),
        val_idx=val_idx, val_scores=val_s, test_idx=test_idx, test_scores=test_s,
    )
    with open(os.path.join(folder, f"{name}.json"), "w") as f:
        json.dump(row, f, indent=2)


def _done(out_dir, family, name) -> bool:
    return os.path.exists(os.path.join(out_dir, "raw", family, f"{name}.json"))


def run_xgb_sweep(X, y, splits, out_dir, variants, levels, n_trials=50, tune_seed=0):
    """Tune on seed 0, then fit and score every seed, for each (fraud rate, variant)."""
    by_key = {(s.seed, s.level): s for s in splits}
    seeds = sorted({s.seed for s in splits})
    params_dir = os.path.join(out_dir, "xgb_params")
    os.makedirs(params_dir, exist_ok=True)

    for level in levels:
        for variant in variants:
            # ---- 1. tuning on seed 0 (skipped if already saved)
            pfile = os.path.join(params_dir, f"{variant}__{level}.json")
            if os.path.exists(pfile):
                best = json.load(open(pfile))
            else:
                s0 = by_key[tune_seed, level]
                print(f"tuning {variant} at fraud rate {level} ...", flush=True)
                best = tune(X[s0.train], y[s0.train], X[s0.val], y[s0.val],
                            variant, n_trials=n_trials, seed=tune_seed)
                best["level"] = level
                json.dump(best, open(pfile, "w"), indent=2)
                print(f"  best validation PR-AUC {best['val_pr_auc']:.4f} "
                      f"in {best['tuning_seconds']:.0f}s", flush=True)

            # ---- 2. refit with those settings on every seed and score on test
            for seed in seeds:
                name = f"xgb_{variant}__{level}__seed{seed}"
                if _done(out_dir, "xgb", name):
                    continue
                s = by_key[seed, level]
                t0 = time.perf_counter()
                model = fit_xgb(X[s.train], y[s.train], X[s.val], y[s.val], best["params"], seed)
                fit_s = time.perf_counter() - t0
                val_s = model.predict_proba(X[s.val])[:, 1]
                t1 = time.perf_counter()
                test_s = model.predict_proba(X[s.test])[:, 1]
                predict_s = time.perf_counter() - t1
                # threshold chosen on validation, then applied unchanged to test
                thr = best_f1_threshold(y[s.val], val_s)
                row = {"family": "xgb", "model": f"XGBoost ({variant})", "variant": variant,
                       "level": level, "seed": seed, "k": len(s.train),
                       "fit_s": round(fit_s, 2), "predict_s": round(predict_s, 3),
                       "n_trees": int(model.best_iteration) + 1, "val_threshold": thr,
                       **all_metrics(y[s.test], test_s, threshold=thr)}
                _save_run(out_dir, "xgb", name, row, s.val, val_s, s.test, test_s)
                print(f"  {name}: test PR-AUC {row['pr_auc']:.4f}", flush=True)


def collect(out_dir, family) -> pd.DataFrame:
    """All saved metric rows for one model family, as one table."""
    folder = os.path.join(out_dir, "raw", family)
    rows = [json.load(open(os.path.join(folder, f)))
            for f in sorted(os.listdir(folder)) if f.endswith(".json")]
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, metrics=("pr_auc", "roc_auc", "recall_at_top1pct",
                                         "precision_at_0.1pct_budget", "ece")) -> pd.DataFrame:
    """Mean and standard deviation over the 5 seeds, the way the proposal reports results."""
    order = ["20", "5", "1", "0.5", "natural"]
    g = df.groupby(["model", "level"])[list(metrics)].agg(["mean", "std"])
    g.columns = [f"{m}_{stat}" for m, stat in g.columns]
    g = g.reset_index()
    g["level"] = pd.Categorical(g["level"], order, ordered=True)
    return g.sort_values(["model", "level"]).reset_index(drop=True)


# ====================================================================== T5: foundation models

def _gpu_peak_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _gpu_peak_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    return float("nan")


def _context_path(out_dir, policy, level, k, seed):
    folder = os.path.join(out_dir, "raw", "contexts")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{policy}__{level}__k{k}__seed{seed}.npy")


def get_context(out_dir, split, y, k, policy, seed):
    """Build the context once and save its row indices, so every model that uses
    this (policy, fraud rate, k, seed) sees exactly the same rows."""
    from .tfm import build_context

    path = _context_path(out_dir, policy, split.level, k, seed)
    if os.path.exists(path):
        saved = np.load(path)
        # Guard against a stale file from an older version of the context code:
        # the rows on Drive must be the rows the current code would pick.
        if not np.array_equal(saved, build_context(split.train, y, k, policy, seed)):
            raise RuntimeError(f"{path} was built by older context code; run "
                               "archive_old_uniform_runs() first")
        return saved
    ctx = build_context(split.train, y, k, policy, seed)
    np.save(path, ctx)
    return ctx


def run_tfm_sweep(X, y, splits, out_dir, model_makers: dict, levels, budgets, seeds,
                  policy="uniform"):
    """Every (model, seed, fraud rate, k) cell for one context policy.

    model_makers maps a display name to a function seed -> unfitted model, e.g.
    {"TabPFN v2": make_tabpfn_v2}. Loop order is model, then seed, so a Colab
    session can finish one model-seed block completely before it runs out.
    """
    from .metrics import all_metrics, best_f1_threshold
    from .tfm import feasible_budgets, predict_in_batches

    by_key = {(s.seed, s.level): s for s in splits}
    for model_name, make in model_makers.items():
        tag = model_name.lower().replace(" ", "")
        for seed in seeds:
            for level in levels:
                s = by_key[seed, level]
                for k in feasible_budgets(s.train, budgets):
                    name = f"{tag}_{policy}__{level}__k{k}__seed{seed}"
                    if _done(out_dir, "tfm", name):
                        continue
                    ctx = get_context(out_dir, s, y, k, policy, seed)
                    n_ctx_fraud = int(y[ctx].sum())
                    _gpu_peak_reset()
                    t0 = time.perf_counter()
                    if n_ctx_fraud == 0:
                        # No fraud in the context: the model has nothing to compare
                        # against. We record it as a model that scores every row the
                        # same, which is what "no information" means for ranking.
                        fit_s = 0.0
                        val_s = np.zeros(len(s.val))
                        t1 = time.perf_counter()
                        test_s = np.zeros(len(s.test))
                    else:
                        model = make(seed)
                        model.fit(X[ctx], y[ctx])
                        fit_s = time.perf_counter() - t0
                        val_s = predict_in_batches(model, X[s.val])
                        t1 = time.perf_counter()
                        test_s = predict_in_batches(model, X[s.test])
                    predict_s = time.perf_counter() - t1
                    thr = best_f1_threshold(y[s.val], val_s)
                    row = {"family": "tfm", "model": model_name, "policy": policy,
                           "level": level, "seed": seed, "k": k,
                           "context_frauds": n_ctx_fraud,
                           "fit_s": round(fit_s, 2), "predict_s": round(predict_s, 2),
                           "seconds_per_1k_test": round(1000 * predict_s / len(s.test), 3),
                           "peak_gpu_gb": _gpu_peak_gb(), "val_threshold": thr,
                           **all_metrics(y[s.test], test_s, threshold=thr)}
                    _save_run(out_dir, "tfm", name, row, s.val, val_s, s.test, test_s)
                    print(f"  {name}: {n_ctx_fraud} frauds in context, "
                          f"test PR-AUC {row['pr_auc']:.4f}, {predict_s:.0f}s", flush=True)


# ====================================================================== T6: matched-budget XGBoost

def run_matched_xgb(X, y, splits, out_dir, levels, budgets, seeds, policy="uniform",
                    variant="plain", n_trials=50, tune_seed=0):
    """XGBoost trained on exactly the k context rows the foundation model saw.

    This separates two things the headline comparison mixes together: the model
    family, and the amount of data each family gets. Full-data XGBoost sees every
    training row; a TFM sees only k of them. Here XGBoost gets the same k.

    Tuning: 50 trials on seed 0's context for each (fraud rate, k), early-stopped on
    that split's validation set, reused for seeds 1-4, the same scheme as T4.
    """
    from .metrics import all_metrics, best_f1_threshold
    from .tfm import feasible_budgets

    by_key = {(s.seed, s.level): s for s in splits}
    params_dir = os.path.join(out_dir, "xgb_matched_params")
    os.makedirs(params_dir, exist_ok=True)
    for level in levels:
        for k in feasible_budgets(by_key[tune_seed, level].train, budgets):
            pfile = os.path.join(params_dir, f"{variant}_{policy}__{level}__k{k}.json")
            if os.path.exists(pfile):
                best = json.load(open(pfile))
            else:
                s0 = by_key[tune_seed, level]
                ctx0 = get_context(out_dir, s0, y, k, policy, tune_seed)
                if y[ctx0].sum() == 0:
                    best = {"params": {}, "note": "no fraud in seed-0 context"}
                else:
                    print(f"tuning matched XGBoost at fraud rate {level}, k={k} ...", flush=True)
                    best = tune(X[ctx0], y[ctx0], X[s0.val], y[s0.val], variant,
                                n_trials=n_trials, seed=tune_seed)
                json.dump(best, open(pfile, "w"), indent=2)
            for seed in seeds:
                name = f"xgb_{variant}_matched_{policy}__{level}__k{k}__seed{seed}"
                if _done(out_dir, "xgb_matched", name):
                    continue
                s = by_key[seed, level]
                ctx = get_context(out_dir, s, y, k, policy, seed)
                n_ctx_fraud = int(y[ctx].sum())
                t0 = time.perf_counter()
                if n_ctx_fraud == 0:
                    fit_s, val_s, test_s = 0.0, np.zeros(len(s.val)), np.zeros(len(s.test))
                else:
                    model = fit_xgb(X[ctx], y[ctx], X[s.val], y[s.val], best["params"], seed)
                    fit_s = time.perf_counter() - t0
                    val_s = model.predict_proba(X[s.val])[:, 1]
                    test_s = model.predict_proba(X[s.test])[:, 1]
                thr = best_f1_threshold(y[s.val], val_s)
                row = {"family": "xgb_matched", "model": f"XGBoost ({variant}, k rows)",
                       "policy": policy, "level": level, "seed": seed, "k": k,
                       "context_frauds": n_ctx_fraud, "fit_s": round(fit_s, 2),
                       "val_threshold": thr, **all_metrics(y[s.test], test_s, threshold=thr)}
                _save_run(out_dir, "xgb_matched", name, row, s.val, val_s, s.test, test_s)
                print(f"  {name}: test PR-AUC {row['pr_auc']:.4f}", flush=True)


def summarise_by_k(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds for each model, fraud rate and context budget k,
    plus how many of the seeds had a context with no fraud at all."""
    order = ["20", "5", "1", "0.5", "natural"]
    g = df.groupby(["model", "level", "k"]).agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        recall_at_top1pct_mean=("recall_at_top1pct", "mean"),
        context_frauds_mean=("context_frauds", "mean"),
        seeds_with_no_fraud=("context_frauds", lambda c: int((c == 0).sum())),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    g["level"] = pd.Categorical(g["level"], order, ordered=True)
    return g.sort_values(["model", "level", "k"]).reset_index(drop=True)


def archive_old_uniform_runs(out_dir) -> bool:
    """One-time move of the first T5/T6 runs (independent context draws per k) out of
    the way, so they are rerun with nested contexts. Nothing is deleted: the old
    files go to <out_dir>/archive_v1_independent_contexts/. Full-data XGBoost
    (raw/xgb, xgb_params) is not touched. Safe to call every session: after the
    first call it finds the marker file and does nothing. Returns True if it moved
    anything."""
    import shutil

    marker = os.path.join(out_dir, "raw", "CONTEXTS_NESTED_V2")
    if os.path.exists(marker):
        return False
    archive = os.path.join(out_dir, "archive_v1_independent_contexts")
    moved = False
    for rel in ["raw/tfm", "raw/xgb_matched", "raw/contexts", "xgb_matched_params"]:
        src = os.path.join(out_dir, rel)
        if os.path.exists(src):
            dst = os.path.join(archive, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved = True
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").write("uniform contexts are nested across k from here on\n")
    return moved


def archive_xgb_search_v1(out_dir) -> bool:
    """One-time move of every XGBoost result tuned with the first search space
    (min_child_weight floor 1) to <out_dir>/archive_v1_xgb_search/, so that both
    full-data and matched-budget XGBoost are retuned with the wider space.
    Foundation-model runs and saved contexts are not touched. Safe to call every
    session; returns True only on the call that moved something."""
    import shutil

    marker = os.path.join(out_dir, "raw", "XGB_SEARCH_V2")
    if os.path.exists(marker):
        return False
    archive = os.path.join(out_dir, "archive_v1_xgb_search")
    moved = False
    for rel in ["raw/xgb", "xgb_params", "raw/xgb_matched", "xgb_matched_params"]:
        src = os.path.join(out_dir, rel)
        if os.path.exists(src):
            dst = os.path.join(archive, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved = True
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").write("XGBoost search space v2: min_child_weight in [1e-3, 20]\n")
    return moved


# ====================================================================== T8: context policies (W1)

def _fit_predict(make, seed, X, y, ctx, *X_queries):
    """Fit one foundation model on a context and score each array in X_queries.
    Returns None if the context has no fraud (the model would have nothing to
    compare against). One fit serves every query array."""
    from .tfm import predict_in_batches

    if y[ctx].sum() == 0:
        return None
    model = make(seed)
    model.fit(X[ctx], y[ctx])
    preds = [predict_in_batches(model, Xq) for Xq in X_queries]
    return preds[0] if len(preds) == 1 else preds


def run_policy_sweep(X, y, splits, out_dir, model_makers: dict, policies, levels, budgets,
                     seeds, m_ensemble=5, ensemble_budgets=None):
    """W1: every context policy, for each (model, seed, fraud rate, k).

    Single-context policies (uniform, stratified, balanced) reuse run_tfm_sweep, so
    they are saved and scored exactly like the T5 runs. kNN and ensemble policies
    need more than one context per run and are handled here. Results go in the
    same "tfm" folder with the policy in the file name, so report.py sees them all.

    ensemble_budgets limits which k the ensembles run at. An ensemble of m contexts
    costs m times a single context, so by default they run at the smallest and the
    largest k only.
    """
    from .metrics import all_metrics, best_f1_threshold
    from .tfm import (GROUP_POLICIES, SINGLE_POLICIES, average_log_odds, ensemble_contexts,
                      feasible_budgets, knn_groups)

    for policy in policies:
        if policy in SINGLE_POLICIES:
            run_tfm_sweep(X, y, splits, out_dir, model_makers, levels, budgets, seeds, policy=policy)
    by_key = {(s.seed, s.level): s for s in splits}
    ens_budgets = ensemble_budgets or [min(budgets), max(budgets)]
    for model_name, make in model_makers.items():
        tag = model_name.lower().replace(" ", "")
        for seed in seeds:
            for level in levels:
                s = by_key[seed, level]
                for policy in policies:
                    if policy in SINGLE_POLICIES:
                        continue
                    ks = feasible_budgets(s.train, budgets)
                    if policy.startswith("ensemble"):
                        ks = [k for k in ks if k in ens_budgets]
                    for k in ks:
                        name = f"{tag}_{policy}__{level}__k{k}__seed{seed}"
                        if _done(out_dir, "tfm", name):
                            continue
                        _gpu_peak_reset()
                        t0 = time.perf_counter()
                        if policy in GROUP_POLICIES:
                            # one context per cluster of query rows, built separately
                            # for validation and test so neither sees the other
                            val_s, test_s = np.zeros(len(s.val)), np.zeros(len(s.test))
                            frauds, rows_no_fraud = [], 0
                            for part, scores in ((s.val, val_s), (s.test, test_s)):
                                for mask, ctx in knn_groups(s.train, X, y, X[part], k, policy, seed):
                                    p = _fit_predict(make, seed, X, y, ctx, X[part][mask])
                                    if part is s.test:
                                        frauds.append(int(y[ctx].sum()))
                                        rows_no_fraud += int(mask.sum()) if p is None else 0
                                    if p is not None:
                                        scores[mask] = p
                            extra = {"context_frauds": float(np.mean(frauds)), "n_contexts": len(frauds),
                                     "test_share_scored_without_fraud": rows_no_fraud / len(s.test)}
                        else:
                            base = policy.split("_", 1)[1]
                            ctxs = ensemble_contexts(s.train, y, k, base, seed, m_ensemble)
                            vals, tests = [], []
                            for ctx in ctxs:
                                p = _fit_predict(make, seed, X, y, ctx, X[s.val], X[s.test])
                                if p is None:
                                    continue   # a context with no fraud adds no information; skip it
                                vals.append(p[0])
                                tests.append(p[1])
                            val_s = average_log_odds(vals) if vals else np.zeros(len(s.val))
                            test_s = average_log_odds(tests) if tests else np.zeros(len(s.test))
                            extra = {"context_frauds": float(np.mean([y[c].sum() for c in ctxs])),
                                     "n_contexts": len(ctxs), "n_contexts_used": len(vals)}
                        elapsed = time.perf_counter() - t0
                        thr = best_f1_threshold(y[s.val], val_s)
                        row = {"family": "tfm", "model": model_name, "policy": policy,
                               "level": level, "seed": seed, "k": k, **extra,
                               "fit_predict_s": round(elapsed, 1), "peak_gpu_gb": _gpu_peak_gb(),
                               "val_threshold": thr, **all_metrics(y[s.test], test_s, threshold=thr)}
                        _save_run(out_dir, "tfm", name, row, s.val, val_s, s.test, test_s)
                        print(f"  {name}: test PR-AUC {row['pr_auc']:.4f}, {elapsed:.0f}s", flush=True)


def run_context_variance(X, y, split, out_dir, make, model_name, k, context_seeds):
    """How much does the answer change if only the context changes?

    The split (train/test rows) is held fixed and the uniform context of size k is
    drawn again with different random seeds. The spread of test PR-AUC across
    these draws is pure context sensitivity. Comparing it with the spread across
    the 5 split seeds (which change both the split and the context) shows how much
    of the run-to-run noise comes from the context alone.
    """
    from .metrics import pr_auc
    from .tfm import predict_in_batches

    tag = model_name.lower().replace(" ", "")
    folder = os.path.join(out_dir, "raw", "context_variance")
    os.makedirs(folder, exist_ok=True)
    rows = []
    for cs in context_seeds:
        path = os.path.join(folder, f"{tag}__{split.level}__k{k}__split{split.seed}__ctx{cs}.json")
        if os.path.exists(path):
            rows.append(json.load(open(path)))
            continue
        # context seed 1000 + cs keeps these draws apart from the seed-0..4 contexts
        ctx = np.sort(np.random.default_rng(1000 + cs).permutation(split.train)[:k])
        model = make(split.seed)
        model.fit(X[ctx], y[ctx])
        s = predict_in_batches(model, X[split.test])
        row = {"model": model_name, "level": split.level, "k": k, "split_seed": split.seed,
               "context_seed": cs, "context_frauds": int(y[ctx].sum()), "pr_auc": pr_auc(y[split.test], s)}
        json.dump(row, open(path, "w"), indent=2)
        rows.append(row)
        print(f"  context {cs}: {row['context_frauds']} frauds, PR-AUC {row['pr_auc']:.4f}", flush=True)
    return pd.DataFrame(rows)
