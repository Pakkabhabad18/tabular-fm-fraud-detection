"""Turning the saved runs into tables and figures (task T7).

Everything here reads the per-run files written by experiments.py and never
re-runs a model, so the tables and figures can be rebuilt at any time on a CPU.

Rule for which context size k represents a foundation model in the headline
comparison: the LARGEST k that fits the training pool at that fraud rate (1K at
20%, 5K at 5%, 10K below that). This is fixed in advance and does not look at
test scores. Picking "the best k" per fraud rate after seeing test PR-AUC would
flatter the foundation models, so it is not done anywhere in this file.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import CONTEXT_BUDGETS, PREVALENCE_LEVELS
from .experiments import collect
from .metrics import paired_bootstrap, pr_auc

LEVEL_ORDER = ["20", "5", "1", "0.5", "natural"]
NATURAL_RATE = 492 / 284_807          # 0.172%, the untouched ULB fraud rate
HEADLINE_XGB = "XGBoost (plain)"      # full-data classical baseline (variants tie, see T4)
MATCHED_XGB = "XGBoost (plain, k rows)"
TFM_MODELS = ["TabICLv2", "TabPFN v2"]
LEVEL_LABEL = {"20": "20%", "5": "5%", "1": "1%", "0.5": "0.5%", "natural": "0.172%"}

# Colours and line styles, fixed per model so a model looks the same in every
# figure. Colours are slots 1-4 of a palette checked for colour-blind safety;
# line style and marker give a second cue so no line relies on colour alone.
STYLE = {
    HEADLINE_XGB: dict(color="#2a78d6", ls="-", marker="o", label="XGBoost, all training rows"),
    "TabICLv2": dict(color="#eb6834", ls="-", marker="s", label="TabICLv2, uniform context"),
    "TabPFN v2": dict(color="#1baf7a", ls="-", marker="^", label="TabPFN v2, uniform context"),
    MATCHED_XGB: dict(color="#eda100", ls="--", marker="D", label="XGBoost, same k rows as the TFM"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fraud_rate(level: str) -> float:
    pi = PREVALENCE_LEVELS[level]
    return NATURAL_RATE if pi is None else pi


# ---------------------------------------------------------------- loading

def load_runs(out_dir) -> pd.DataFrame:
    """One row per saved run (model x fraud rate x k x seed), all families together."""
    parts = []
    for family in ("xgb", "tfm", "xgb_matched"):
        if os.path.exists(os.path.join(out_dir, "raw", family)):
            parts.append(collect(out_dir, family))
    df = pd.concat(parts, ignore_index=True)
    df["level"] = pd.Categorical(df["level"].astype(str), LEVEL_ORDER, ordered=True)
    return df


def largest_k(level: str, train_rows: dict) -> int:
    """Largest context budget that fits the training pool at this fraud rate."""
    return max(k for k in CONTEXT_BUDGETS if k <= train_rows[level])


# ---------------------------------------------------------------- tables

def headline_table(runs: pd.DataFrame, train_rows: dict) -> pd.DataFrame:
    """Mean and std of PR-AUC over seeds, per model and fraud rate.

    Full-data XGBoost uses all training rows. Each TFM and the matched XGBoost
    use the largest k at that fraud rate (see the module note).
    """
    rows = []
    for level in [l for l in LEVEL_ORDER if l in train_rows]:   # only rates that were run
        k = largest_k(level, train_rows)
        for model in [HEADLINE_XGB, *TFM_MODELS, MATCHED_XGB]:
            sel = runs[(runs.model == model) & (runs.level == level)]
            if model != HEADLINE_XGB:
                sel = sel[sel.k == k]
            if sel.empty:
                continue
            rows.append({
                "level": level, "fraud_rate": fraud_rate(level), "model": model,
                "k": int(sel.k.iloc[0]) if model != HEADLINE_XGB else int(train_rows[level]),
                "n_seeds": sel.seed.nunique(),
                "pr_auc_mean": sel.pr_auc.mean(), "pr_auc_std": sel.pr_auc.std(ddof=1),
                "roc_auc_mean": sel.roc_auc.mean(),
                "context_frauds_mean": sel["context_frauds"].mean() if "context_frauds" in sel else np.nan,
            })
    return pd.DataFrame(rows)


def by_k_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of PR-AUC for every (model, fraud rate, k) cell."""
    sub = runs[runs.model.isin(TFM_MODELS + [MATCHED_XGB])]
    return (sub.groupby(["model", "level", "k"], observed=True)
               .agg(n_seeds=("seed", "nunique"), pr_auc_mean=("pr_auc", "mean"),
                    pr_auc_std=("pr_auc", "std"), context_frauds_mean=("context_frauds", "mean"),
                    seeds_with_no_fraud=("context_frauds", lambda c: int((c == 0).sum())))
               .reset_index())


def paired_tests(out_dir, y, runs: pd.DataFrame, train_rows: dict, n_boot=1000) -> pd.DataFrame:
    """Is each TFM's PR-AUC really different from full-data XGBoost, seed by seed?

    For every seed both models were scored on exactly the same test rows, so the
    paired bootstrap from metrics.py applies directly. We report, per fraud rate:
    the mean difference (TFM minus XGBoost) and in how many of the 5 seeds the 95%
    interval of the difference lies entirely below zero, entirely above zero, or
    crosses zero (no detectable difference on that seed).
    """
    rows = []
    for level in [l for l in LEVEL_ORDER if l in train_rows]:   # only rates that were run
        k = largest_k(level, train_rows)
        for model in TFM_MODELS + [MATCHED_XGB]:
            res = []
            for seed in sorted(runs.seed.unique()):
                a = _scores(out_dir, model, level, k, seed)
                b = _scores(out_dir, HEADLINE_XGB, level, None, seed)
                if a is None or b is None:
                    continue
                assert np.array_equal(a[0], b[0]), "test rows differ between models"
                res.append(paired_bootstrap(y[a[0]], a[1], b[1], metric=pr_auc,
                                            n_boot=n_boot, seed=seed))
            if res:
                rows.append({
                    "level": level, "model": model, "k": k, "n_seeds": len(res),
                    "mean_diff_vs_xgb": float(np.mean([r["diff"] for r in res])),
                    "seeds_tfm_worse": sum(r["ci_high"] < 0 for r in res),
                    "seeds_tfm_better": sum(r["ci_low"] > 0 for r in res),
                    "seeds_no_clear_difference": sum(r["ci_low"] <= 0 <= r["ci_high"] for r in res),
                })
    return pd.DataFrame(rows)


def _scores(out_dir, model, level, k, seed):
    """Test row indices and scores of one saved run, or None if it isn't there."""
    if model == HEADLINE_XGB:
        path = os.path.join(out_dir, "raw", "xgb", f"xgb_plain__{level}__seed{seed}.npz")
    elif model == MATCHED_XGB:
        path = os.path.join(out_dir, "raw", "xgb_matched",
                            f"xgb_plain_matched_uniform__{level}__k{k}__seed{seed}.npz")
    else:
        tag = model.lower().replace(" ", "")
        path = os.path.join(out_dir, "raw", "tfm", f"{tag}_uniform__{level}__k{k}__seed{seed}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["test_idx"], d["test_scores"]


# ---------------------------------------------------------------- figures

def _style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_headline(table: pd.DataFrame, path: str):
    """Figure 1: PR-AUC against fraud rate, both families, +-1 std over seeds.

    The x axis runs from 20% fraud down to 0.172%, on a log scale, so each step
    to the right means fraud is rarer. The grey dotted line is what a model with
    no skill scores (PR-AUC equals the fraud rate).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for model in [HEADLINE_XGB, *TFM_MODELS, MATCHED_XGB]:
        t = table[table.model == model].sort_values("fraud_rate", ascending=False)
        if t.empty:
            continue
        st = STYLE[model]
        x, m, s = t.fraud_rate.values, t.pr_auc_mean.values, np.nan_to_num(t.pr_auc_std.values)
        ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.12, lw=0)
        ax.plot(x, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=6, lw=2,
                label=st["label"], markeredgecolor="white", markeredgewidth=1)
    rates = np.array(sorted(table.fraud_rate.unique(), reverse=True))
    ax.plot(rates, rates, color=MUTED, ls=":", lw=1.2, label="No skill (PR-AUC = fraud rate)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(rates)
    levels_sorted = table.drop_duplicates("fraud_rate").sort_values("fraud_rate", ascending=False)
    ax.set_xticklabels([LEVEL_LABEL[str(l)] for l in levels_sorted.level])
    ax.minorticks_off()
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Fraud rate in train and test (log scale, rarer to the right)", color=INK, fontsize=10)
    ax.set_ylabel("Test PR-AUC (mean over seeds)", color=INK, fontsize=10)
    ax.set_title("PR-AUC as fraud gets rarer: all-data XGBoost vs foundation models at their largest k",
                 color=INK, fontsize=10.5, loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_by_k(by_k: pd.DataFrame, xgb_full: pd.DataFrame, path: str):
    """Figure 2: PR-AUC against context size k, one panel per fraud rate.

    The flat blue line in each panel is all-data XGBoost at that fraud rate, the
    level a foundation model would have to reach.
    """
    import matplotlib.pyplot as plt

    levels = [l for l in LEVEL_ORDER if l in set(by_k.level.astype(str))]
    fig, axes = plt.subplots(1, len(levels), figsize=(2.6 * len(levels), 3.3), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        for model in [*TFM_MODELS, MATCHED_XGB]:
            t = by_k[(by_k.model == model) & (by_k.level.astype(str) == level)].sort_values("k")
            if t.empty:
                continue
            st = STYLE[model]
            m, s = t.pr_auc_mean.values, np.nan_to_num(t.pr_auc_std.values)
            ax.fill_between(t.k, m - s, m + s, color=st["color"], alpha=0.12, lw=0)
            ax.plot(t.k, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.8,
                    label=st["label"], markeredgecolor="white", markeredgewidth=0.8)
        ref = xgb_full[(xgb_full.model == HEADLINE_XGB) & (xgb_full.level.astype(str) == level)]
        if not ref.empty:
            ax.axhline(ref.pr_auc.mean(), color=STYLE[HEADLINE_XGB]["color"], lw=1.5,
                       label=STYLE[HEADLINE_XGB]["label"])
        ax.set_xscale("log")
        ax.set_xticks(CONTEXT_BUDGETS)
        ax.set_xticklabels([f"{k // 1000}K" for k in CONTEXT_BUDGETS])
        ax.minorticks_off()
        ax.set_xlim(800, 12_500)
        ax.set_title(f"fraud rate {LEVEL_LABEL[level]}", color=INK, fontsize=9.5)
        ax.set_xlabel("context size k", color=INK, fontsize=9)
        _style_axes(ax)
    axes[0].set_ylabel("Test PR-AUC", color=INK, fontsize=9.5)
    axes[0].set_ylim(0, 1.0)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_vs_context_frauds(runs: pd.DataFrame, path: str, level: str = "natural"):
    """Figure 3: PR-AUC against how many frauds the context actually held.

    One point per (k, seed) run at the natural fraud rate. This separates the two
    families most clearly: the foundation models score well with a handful of
    frauds, while XGBoost trained on the same rows needs more examples.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    sub = runs[runs.level.astype(str) == level]
    jitter = {"TabICLv2": 0.94, "TabPFN v2": 1.0, MATCHED_XGB: 1.06}   # nudge so points don't overlap
    for model in [*TFM_MODELS, MATCHED_XGB]:
        t = sub[sub.model == model]
        if t.empty:
            continue
        st = STYLE[model]
        x = np.maximum(t.context_frauds.values, 0.5) * jitter[model]   # 0 frauds drawn at 0.5
        ax.scatter(x, t.pr_auc, s=34, color=st["color"], marker=st["marker"],
                   edgecolor="white", linewidth=0.8, label=st["label"], zorder=3)
    ax.axhline(fraud_rate(level), color=MUTED, ls=":", lw=1.2, label="No skill")
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("Frauds in the context (log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Test PR-AUC", color=INK, fontsize=10)
    ax.set_title("At the natural 0.172% rate: score against frauds seen", color=INK,
                 fontsize=10.5, loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
