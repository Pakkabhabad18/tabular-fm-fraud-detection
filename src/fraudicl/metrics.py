"""Scoring code shared by both model families (task T3).

Every model hands in the same two things: the true labels y (1 = fraud) and a
score per row (higher = more likely fraud; for most models this is the predicted
probability). Every metric in the proposal is computed from those two arrays here,
so the families cannot end up scored in slightly different ways.

Why no accuracy: predicting "not fraud" for every row already scores 99.83% on
ULB. All metrics below look at how well frauds are ranked above legitimate rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ---------------------------------------------------------------- ranking metrics

def pr_auc(y, s) -> float:
    """Area under the precision-recall curve (average precision). Headline metric.

    A model that guesses at random scores about the fraud rate (0.0017 at 0.172%),
    so PR-AUC is only meaningful next to the fraud rate of the test set it came from.
    """
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    """Area under the ROC curve. Reported for comparison with published work only:
    it stays high under heavy imbalance even when most flagged rows are false alarms."""
    return float(roc_auc_score(y, s))


def _top_k_mask(s, k: int) -> np.ndarray:
    """True for the k highest-scoring rows. Ties are broken by row order (stable)."""
    order = np.argsort(-np.asarray(s), kind="stable")
    mask = np.zeros(len(s), dtype=bool)
    mask[order[:k]] = True
    return mask


def recall_at_top_fraction(y, s, frac: float = 0.01) -> float:
    """Share of all frauds caught if analysts review only the top `frac` of rows.

    Default 1%, as in the proposal: flag the riskiest 1% of transactions and ask
    how many of the frauds are among them.
    """
    y = np.asarray(y)
    k = max(1, int(np.ceil(frac * len(y))))
    return float(y[_top_k_mask(s, k)].sum() / y.sum())


def precision_at_budget(y, s, frac: float = 0.001) -> float:
    """Share of flagged rows that are really fraud, at an alert budget of `frac`.

    Default 0.1%, as in the proposal: about the number of cards a review team can
    actually check per day. This is what an analyst feels: how many alerts are wasted.
    """
    y = np.asarray(y)
    k = max(1, int(np.ceil(frac * len(y))))
    return float(y[_top_k_mask(s, k)].mean())


def recall_at_fpr(y, s, max_fpr: float = 0.001) -> float:
    """Highest share of frauds caught while wrongly flagging at most `max_fpr` of
    legitimate rows (default 0.1%). Read off the ROC curve."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= max_fpr].max())


# ---------------------------------------------------------------- thresholds and F1

def f1_at_threshold(y, s, threshold: float) -> float:
    """F1 score when every row with score >= threshold is flagged as fraud."""
    y = np.asarray(y)
    flagged = np.asarray(s) >= threshold
    tp = int((flagged & (y == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / flagged.sum()
    recall = tp / (y == 1).sum()
    return float(2 * precision * recall / (precision + recall))


def best_f1_threshold(y_val, s_val) -> float:
    """The threshold that maximises F1 on the VALIDATION set.

    It is then applied unchanged to the test set. Choosing the threshold on test
    would make F1 look better than it would be in real use.
    """
    candidates = np.unique(np.asarray(s_val))
    scores = [f1_at_threshold(y_val, s_val, t) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


# ---------------------------------------------------------------- calibration

def expected_calibration_error(y, p, n_bins: int = 15, strategy: str = "quantile") -> float:
    """Average gap between predicted fraud probability and the observed fraud rate.

    Rows are grouped into n_bins by predicted probability. In each bin we compare
    the mean prediction with the actual share of frauds, and weight by bin size.

    strategy="quantile" (default) puts the same number of rows in each bin. With
    0.172% fraud almost every prediction is near 0, so equal-width bins would put
    nearly all rows in the first bin and hide the errors that matter.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    # assign each row to a bin; the last edge is included in the last bin
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


# ---------------------------------------------------------------- one call for everything

def all_metrics(y, s, threshold: float | None = None) -> dict:
    """Every proposal metric for one model on one test set.

    `threshold` should come from best_f1_threshold on the validation set. If it is
    left out, F1 is skipped. ECE assumes s is a probability in [0, 1].
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    out = {
        "n_rows": int(len(y)),
        "n_frauds": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        "recall_at_top1pct": recall_at_top_fraction(y, s, 0.01),
        "precision_at_0.1pct_budget": precision_at_budget(y, s, 0.001),
        "recall_at_fpr_0.1pct": recall_at_fpr(y, s, 0.001),
    }
    if threshold is not None:
        out["f1_at_val_threshold"] = f1_at_threshold(y, s, threshold)
    if s.min() >= 0 and s.max() <= 1:
        out["ece"] = expected_calibration_error(y, s)
    return out


# ---------------------------------------------------------------- paired bootstrap

def paired_bootstrap(y, s_a, s_b, metric=pr_auc, n_boot: int = 1000, seed: int = 0) -> dict:
    """Is model A really better than model B on this test set, or is it noise?

    Both models are scored on the same resampled rows each time (that is what
    "paired" means), and we look at the spread of metric(A) - metric(B).

    The resampling is stratified: frauds and legitimate rows are resampled
    separately, so every resample keeps exactly the same number of frauds. With
    only about 98 test frauds, a plain bootstrap would change the fraud count from
    resample to resample, and PR-AUC moves with the fraud rate, which adds noise
    that has nothing to do with the models.

    Returns the observed difference, a 95% interval, and a two-sided p-value
    (how often the resampled difference falls on the other side of zero).
    """
    y = np.asarray(y)
    s_a = np.asarray(s_a)
    s_b = np.asarray(s_b)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[i] = metric(y[idx], s_a[idx]) - metric(y[idx], s_b[idx])
    observed = metric(y, s_a) - metric(y, s_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two_sided = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_two_sided),
    }
