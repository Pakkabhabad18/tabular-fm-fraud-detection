"""Checks for the scoring code (task T3) on toy cases where the answer is known.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest

from fraudicl import metrics as M


@pytest.fixture
def toy():
    # 10 rows, 2 frauds. The model ranks one fraud first and the other 4th.
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    return y, s


def test_pr_auc_by_hand(toy):
    # Frauds at ranks 1 and 4: precision there is 1/1 and 2/4.
    # Average precision = (1.0 + 0.5) / 2 = 0.75.
    y, s = toy
    assert M.pr_auc(y, s) == pytest.approx(0.75)


def test_perfect_model_scores_one():
    y = np.array([0, 0, 1, 0, 1])
    s = y.astype(float)
    assert M.pr_auc(y, s) == 1.0
    assert M.roc_auc(y, s) == 1.0
    assert M.recall_at_fpr(y, s, 0.0) == 1.0


def test_random_model_pr_auc_near_fraud_rate():
    # A model with no skill scores about the fraud rate, not 0.5.
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.01).astype(int)
    s = rng.random(200_000)
    assert M.pr_auc(y, s) == pytest.approx(0.01, abs=0.002)
    assert M.roc_auc(y, s) == pytest.approx(0.5, abs=0.02)


def test_recall_and_precision_at_top(toy):
    y, s = toy
    # top 20% = 2 rows (ranks 1-2): 1 of 2 frauds caught, 1 of 2 flags correct
    assert M.recall_at_top_fraction(y, s, 0.2) == 0.5
    assert M.precision_at_budget(y, s, 0.2) == 0.5
    # top 40% = 4 rows: both frauds caught
    assert M.recall_at_top_fraction(y, s, 0.4) == 1.0


def test_recall_at_fpr(toy):
    y, s = toy
    # With no false alarms allowed, only the top fraud (score 0.9) is caught.
    assert M.recall_at_fpr(y, s, 0.0) == 0.5
    # Allowing 2 of 8 legitimate rows (25%) reaches the second fraud at rank 4.
    assert M.recall_at_fpr(y, s, 0.25) == 1.0


def test_f1_and_threshold(toy):
    y, s = toy
    # Flag score >= 0.6: 4 rows flagged, 2 are frauds -> P = 0.5, R = 1 -> F1 = 2/3
    assert M.f1_at_threshold(y, s, 0.6) == pytest.approx(2 / 3)
    # Flag only 0.9: P = 1, R = 0.5 -> F1 = 2/3 as well; the best F1 is 2/3
    assert M.f1_at_threshold(y, s, 0.9) == pytest.approx(2 / 3)
    t = M.best_f1_threshold(y, s)
    assert M.f1_at_threshold(y, s, t) == pytest.approx(2 / 3)


def test_ece_zero_when_calibrated():
    # Predictions that equal the true fraud rate in every bin -> ECE = 0.
    p = np.repeat([0.1, 0.5], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)]])
    assert M.expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.0, abs=1e-9)


def test_ece_catches_overconfidence():
    # Model says 0.5 for everything, true fraud rate is 0.1 -> ECE = 0.4.
    p = np.full(1000, 0.5)
    y = np.r_[np.ones(100), np.zeros(900)]
    assert M.expected_calibration_error(y, p, n_bins=5, strategy="uniform") == pytest.approx(0.4)


def test_all_metrics_keys(toy):
    y, s = toy
    out = M.all_metrics(y, s, threshold=0.6)
    for key in ["pr_auc", "roc_auc", "recall_at_top1pct", "precision_at_0.1pct_budget",
                "recall_at_fpr_0.1pct", "f1_at_val_threshold", "ece"]:
        assert key in out


def test_bootstrap_same_model_is_not_significant():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    s = rng.random(5000) + y  # a decent model
    r = M.paired_bootstrap(y, s, s, n_boot=200)
    assert r["diff"] == 0 and r["p_value"] == 1.0


def test_bootstrap_detects_clear_gap():
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.02).astype(int)
    good = rng.random(5000) + 2 * y
    bad = rng.random(5000)
    r = M.paired_bootstrap(y, good, bad, n_boot=200)
    assert r["diff"] > 0 and r["ci_low"] > 0 and r["p_value"] < 0.01


def test_bootstrap_keeps_fraud_count(monkeypatch):
    # Every resample must contain exactly the same number of frauds as the test set.
    y = np.r_[np.ones(7), np.zeros(93)].astype(int)
    s = np.linspace(0, 1, 100)
    seen = []

    def spy(yy, ss):
        seen.append(int(yy.sum()))
        return M.pr_auc(yy, ss)

    M.paired_bootstrap(y, s, s, metric=spy, n_boot=20)
    assert set(seen) == {7}
