"""W4: how the two families scale with data, and how fast they predict.

Two questions from the proposal:

1. Dataset scale. At the natural fraud rate, XGBoost keeps improving as it gets
   more training rows, while a foundation model only sees its context. How does
   XGBoost's PR-AUC grow from 1K rows to all 170,883 training rows, and where on
   that curve does TabICLv2 sit with 1K-10K context rows, and with 20K and 50K?
   (TabPFN v2 stops at 10K rows; TabICLv2 was pretrained on tables up to 60K.)

2. Latency. A card authorisation has to be scored in milliseconds. How long does
   each model take to score one transaction, and a batch of 1,000?

The training rows for every size N are the first N rows of the same per-seed
random order the uniform contexts use, so the subsets are nested and the 10K
subset is exactly the uniform 10K context.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .experiments import _done, _save_run
from .metrics import all_metrics, best_f1_threshold
from .xgb import fit_xgb


def scale_rows(train_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """First n rows of the seed's uniform order (the same order as build_context's
    uniform policy), so subsets are nested across n."""
    order = np.random.default_rng(seed).permutation(train_idx)
    return np.sort(order[:n])


def run_xgb_scale(X, y, splits, out_dir, sizes, seeds, params: dict, level="natural"):
    """XGBoost (plain) trained on N training rows, for every N in sizes and seed.

    The settings are the ones tuned on the full training set at this fraud rate
    (notebook 02), reused for every N, so the curve shows the effect of data alone.
    Early stopping still uses each seed's validation set, so smaller N gets fewer
    trees if it needs them.
    """
    by_key = {(s.seed, s.level): s for s in splits}
    for seed in seeds:
        s = by_key[seed, level]
        for n in [n for n in sizes if n <= len(s.train)]:
            name = f"xgb_plain_scale__{level}__n{n}__seed{seed}"
            if _done(out_dir, "xgb_scale", name):
                continue
            rows = scale_rows(s.train, n, seed)
            t0 = time.perf_counter()
            if y[rows].sum() == 0:
                val_s, test_s, n_trees = np.zeros(len(s.val)), np.zeros(len(s.test)), 0
            else:
                model = fit_xgb(X[rows], y[rows], X[s.val], y[s.val], params, seed)
                val_s = model.predict_proba(X[s.val])[:, 1]
                test_s = model.predict_proba(X[s.test])[:, 1]
                n_trees = int(model.best_iteration) + 1
            fit_predict_s = time.perf_counter() - t0
            thr = best_f1_threshold(y[s.val], val_s)
            row = {"family": "xgb_scale", "model": "XGBoost (plain, N rows)", "policy": "scale",
                   "level": level, "seed": seed, "k": n, "context_frauds": int(y[rows].sum()),
                   "n_trees": n_trees, "fit_predict_s": round(fit_predict_s, 2),
                   "val_threshold": thr, **all_metrics(y[s.test], test_s, threshold=thr)}
            _save_run(out_dir, "xgb_scale", name, row, s.val, val_s, s.test, test_s)
            print(f"  {name}: {row['context_frauds']} frauds, test PR-AUC {row['pr_auc']:.4f}", flush=True)


def scale_table(runs: pd.DataFrame, level="natural") -> pd.DataFrame:
    """PR-AUC and F1 against the number of rows each model learned from, mean and
    std over seeds: XGBoost on N rows, TabICLv2 and TabPFN v2 with a uniform
    context of k rows (including the 20K and 50K TabICLv2 runs)."""
    keep = (runs.level.astype(str) == level) & (
        (runs.family == "xgb_scale")
        | ((runs.family == "tfm") & runs.policy.isin(["uniform", "uniform_large"])))
    sub = runs[keep].copy()
    sub["rows_seen"] = sub["k"].astype(int)
    return (sub.groupby(["model", "rows_seen"])
               .agg(n_seeds=("seed", "nunique"), pr_auc_mean=("pr_auc", "mean"),
                    pr_auc_std=("pr_auc", "std"), f1_mean=("f1_at_val_threshold", "mean"),
                    frauds_seen=("context_frauds", "mean"))
               .reset_index())


def plot_scale(table: pd.DataFrame, path: str, full_xgb: float | None = None):
    """Figure 7: PR-AUC against the number of training/context rows, log scale."""
    import matplotlib.pyplot as plt

    from .report import INK, MUTED, STYLE, _style_axes

    styles = {"XGBoost (plain, N rows)": dict(color=STYLE["XGBoost (plain)"]["color"], marker="o",
                                              label="XGBoost trained on N rows"),
              "TabICLv2": dict(color=STYLE["TabICLv2"]["color"], marker="s",
                               label="TabICLv2, uniform context of N rows"),
              "TabPFN v2": dict(color=STYLE["TabPFN v2"]["color"], marker="^",
                                label="TabPFN v2, uniform context of N rows")}
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for model, st in styles.items():
        t = table[table.model == model].sort_values("rows_seen")
        if t.empty:
            continue
        m, s = t.pr_auc_mean.values, np.nan_to_num(t.pr_auc_std.values)
        ax.fill_between(t.rows_seen, m - s, m + s, color=st["color"], alpha=0.12, lw=0)
        ax.plot(t.rows_seen, m, color=st["color"], marker=st["marker"], ms=6, lw=2,
                label=st["label"], markeredgecolor="white", markeredgewidth=1)
    if full_xgb is not None:
        ax.axhline(full_xgb, color=MUTED, ls=":", lw=1.2, label="XGBoost, all 170,883 rows")
    ax.set_xscale("log")
    ticks = sorted(table.rows_seen.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{n // 1000}K" if n < 170_000 else "171K" for n in ticks], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Rows the model learns from (training rows or context rows, log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Test PR-AUC at 0.172% (mean over seeds)", color=INK, fontsize=10)
    ax.set_title("More data: XGBoost keeps climbing; a context is capped", color=INK, fontsize=10.5, loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- latency

def latency_table(models: dict, X_query: np.ndarray, batch_sizes=(1, 100, 1_000, 10_000),
                  repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Wall-clock time to score a batch of transactions, for each model and batch size.

    models: name -> fitted model with predict_proba. Each (model, batch size) is
    timed `repeats` times on different random rows after one warm-up call (the
    first call pays one-off costs such as moving weights to the GPU); the median
    is reported. predict_proba returns a numpy array, so GPU work has finished when
    the timer stops.

    For the foundation models, every call processes the whole context again, so
    even one transaction costs about as much as the context itself. That is the
    number to compare with the few milliseconds an authorisation can wait.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, model in models.items():
        for b in batch_sizes:
            if b > len(X_query):
                continue
            model.predict_proba(X_query[rng.choice(len(X_query), b, replace=False)])   # warm-up
            times = []
            for _ in range(repeats):
                q = X_query[rng.choice(len(X_query), b, replace=False)]
                t0 = time.perf_counter()
                model.predict_proba(q)
                times.append(time.perf_counter() - t0)
            med = float(np.median(times))
            rows.append({"model": name, "batch_size": b, "seconds_per_call": med,
                         "ms_per_transaction": 1000 * med / b, "transactions_per_second": b / med})
            print(f"  {name}, batch {b}: {med * 1000:.1f} ms per call", flush=True)
    return pd.DataFrame(rows)
