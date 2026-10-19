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
