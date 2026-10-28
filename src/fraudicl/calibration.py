"""W2: are the predicted probabilities trustworthy, and where should the alarm
threshold go? (Research question 3.)

Everything here works on scores already saved by the experiment notebooks, so it
runs on a CPU and never refits a model.

Three ideas, in order:

1. Prior correction. A foundation model's probabilities reflect the fraud rate of
   its CONTEXT, not of the data it will be used on. A balanced context at the
   natural rate holds ~3% fraud (295 of 10K rows) against 0.172% in reality, so
   the model thinks fraud is ~17x more common than it is and its probabilities
   come out too high. Bayes' rule removes this exactly when only the class
   balance differs (label shift):

       p' = p * r1 / (p * r1 + (1 - p) * r0),  r1 = pi_true / pi_ctx,
                                               r0 = (1 - pi_true) / (1 - pi_ctx)

   The correction never changes the order of the scores, so PR-AUC, ROC-AUC and
   every top-k metric stay exactly the same. Only calibration (ECE, Brier) and
   fixed thresholds such as 0.5 or pi_1 are affected.

2. Calibration: ECE and Brier score before and after the correction, plus the
   points for a reliability diagram.

3. Threshold rules, compared on the same test rows:
     tau = 0.5        the default a library uses when you call predict()
     tau = pi_1       McDowell et al. (2026): flag a row when its fraud probability
                      exceeds the base fraud rate; tested by them down to 5%
     tau = F1-best    the threshold with the best F1 on VALIDATION, then frozen
     top 0.1%         an alert budget: flag the 0.1% highest-scoring rows,
                      roughly what a review team can check per day
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .config import PREVALENCE_LEVELS
from .metrics import best_f1_threshold, expected_calibration_error

NATURAL_RATE = 492 / 284_807


def true_rate(level: str) -> float:
    """The fraud rate the model will actually face at this level (train = test)."""
    pi = PREVALENCE_LEVELS[str(level)]
    return NATURAL_RATE if pi is None else pi


def prior_correct(p, pi_ctx: float, pi_true: float) -> np.ndarray:
    """Rescale probabilities learned at fraud rate pi_ctx to fraud rate pi_true."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    r1 = pi_true / pi_ctx
    r0 = (1 - pi_true) / (1 - pi_ctx)
    return p * r1 / (p * r1 + (1 - p) * r0)


def threshold_report(y_val, p_val, y_test, p_test, pi_true: float) -> list[dict]:
    """Precision, recall and F1 on test for each threshold rule (see module note)."""
    y_test = np.asarray(y_test)
    p_test = np.asarray(p_test)
    # Alert budget: exactly the 0.1% highest-scoring rows, picked by rank (ties broken
    # by row order, as in metrics.precision_at_budget), so tied scores can't push the
    # number of alerts over budget.
    n_budget = max(1, int(np.ceil(0.001 * len(y_test))))
    top = np.zeros(len(p_test), dtype=bool)
    top[np.argsort(-p_test, kind="stable")[:n_budget]] = True
    rules = {
        "tau = 0.5": 0.5,
        "tau = pi_1": pi_true,
        "tau = F1-best on validation": best_f1_threshold(y_val, p_val),
        "top 0.1% alert budget": None,
    }
    out = []
    for name, tau in rules.items():
        flagged = top if tau is None else p_test >= tau
        if tau is None:
            tau = float(p_test[top].min())   # reported for reference only
        tp = int((flagged & (y_test == 1)).sum())
        n_flag = int(flagged.sum())
        prec = tp / n_flag if n_flag else 0.0
        rec = tp / int(y_test.sum())
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        out.append({"rule": name, "threshold": float(tau), "n_flagged": n_flag,
                    "true_frauds_caught": tp, "precision": prec, "recall": rec, "f1": f1})
    return out


def reliability_points(y, p, n_bins: int = 15) -> pd.DataFrame:
    """Mean predicted probability vs observed fraud rate in equal-count bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return pd.DataFrame({"mean_predicted": [p[b].mean() for b in bins],
                         "observed_rate": [y[b].mean() for b in bins],
                         "n_rows": [len(b) for b in bins]})


def _context_rate(row: dict) -> float | None:
    """Fraud share of the context a run used. Full-data XGBoost was trained on the
    split's own fraud rate, so it needs no correction (returns None)."""
    if row.get("family") == "xgb" or not row.get("k"):
        return None
    return max(row["context_frauds"], 0.5) / row["k"]   # 0.5: avoid dividing by zero


def analyse_run(json_path: str, y: np.ndarray) -> tuple[dict, list[dict]]:
    """Calibration and threshold results for one saved run (raw and corrected)."""
    row = json.load(open(json_path))
    d = np.load(json_path.replace(".json", ".npz"))
    y_val, y_test = y[d["val_idx"]], y[d["test_idx"]]
    p_val, p_test = d["val_scores"], d["test_scores"]
    pi_true = true_rate(row["level"])
    pi_ctx = _context_rate(row)
    if pi_ctx is not None:
        c_val, c_test = prior_correct(p_val, pi_ctx, pi_true), prior_correct(p_test, pi_ctx, pi_true)
    else:
        c_val, c_test = p_val, p_test
    key = {k: row.get(k) for k in ("model", "policy", "level", "k", "seed")}
    calib = {**key, "pi_true": pi_true, "pi_context": pi_ctx,
             "mean_pred_raw": float(p_test.mean()), "mean_pred_corrected": float(c_test.mean()),
             "observed_rate": float(y_test.mean()),
             "ece_raw": expected_calibration_error(y_test, p_test),
             "ece_corrected": expected_calibration_error(y_test, c_test),
             "brier_raw": brier_score_loss(y_test, p_test),
             "brier_corrected": brier_score_loss(y_test, c_test)}
    thresholds = [{**key, "probabilities": "corrected", **r}
                  for r in threshold_report(y_val, c_val, y_test, c_test, pi_true)]
    thresholds += [{**key, "probabilities": "raw", **r}
                   for r in threshold_report(y_val, p_val, y_test, p_test, pi_true)]
    return calib, thresholds


def analyse_all(out_dir: str, y: np.ndarray, families=("xgb", "tfm", "xgb_matched"),
                xgb_variants=("plain",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run analyse_run over every saved run. kNN runs are skipped: their contexts
    differ per group of rows, so one context fraud rate doesn't describe them.
    Full-data XGBoost uses the plain variant only (SMOTE and scale_pos_weight change
    the effective training prior in ways a single rate doesn't capture)."""
    calib, thresh = [], []
    for family in families:
        folder = os.path.join(out_dir, "raw", family)
        if not os.path.exists(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".json"):
                continue
            meta = json.load(open(os.path.join(folder, f)))
            if str(meta.get("policy", "")).startswith("knn"):
                continue
            if family == "xgb" and meta.get("variant") not in xgb_variants:
                continue
            c, t = analyse_run(os.path.join(folder, f), y)
            calib.append(c)
            thresh += t
    return pd.DataFrame(calib), pd.DataFrame(thresh)


def plot_reliability(curves: dict, path: str, title: str):
    """Figure 5: reliability diagram. curves maps a label to a reliability_points
    table. Points on the diagonal are perfectly calibrated; points below it mean the
    model says fraud is more likely than it really is. Both axes are log scale
    because almost all predictions at 0.172% fraud are tiny."""
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    markers = ["o", "s", "^", "D", "X"]
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=150)
    lo = 1e-5
    ax.plot([lo, 1], [lo, 1], color="#52514e", ls=":", lw=1.2, label="Perfect calibration")
    for (label, pts), c, m in zip(curves.items(), colors, markers):
        x = np.clip(pts.mean_predicted, lo, 1)
        yv = np.clip(pts.observed_rate, lo, 1)
        ax.plot(x, yv, color=c, marker=m, ms=5, lw=1.6, label=label,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel("Mean predicted fraud probability (bin)", fontsize=10)
    ax.set_ylabel("Observed fraud rate (bin)", fontsize=10)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.grid(True, color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
