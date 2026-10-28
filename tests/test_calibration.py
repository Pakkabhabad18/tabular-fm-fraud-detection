"""Checks for W2 calibration and thresholds.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from fraudicl import calibration as C
from fraudicl.data import make_splits
from fraudicl.experiments import run_policy_sweep, run_xgb_sweep


def test_prior_correction_is_exact_under_label_shift():
    # A model that knows the true posterior at a 50% prior: correcting to a 1% prior
    # must give the posterior at 1%, worked out by hand with Bayes' rule.
    p50 = np.array([0.5, 0.9, 0.1])
    lr = p50 / (1 - p50)                         # likelihood ratio implied at a 50% prior
    expected = lr * 0.01 / (lr * 0.01 + 0.99)    # posterior at a 1% prior
    assert np.allclose(C.prior_correct(p50, 0.5, 0.01), expected)


def test_correction_keeps_ranking():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    p = rng.random(5000) * 0.5 + 0.4 * y
    q = C.prior_correct(p, 0.3, 0.0017)
    assert np.array_equal(np.argsort(p), np.argsort(q))
    assert average_precision_score(y, p) == pytest.approx(average_precision_score(y, q))


def test_correction_identity_when_priors_match():
    p = np.linspace(0.01, 0.99, 50)
    assert np.allclose(C.prior_correct(p, 0.05, 0.05), p)


def test_threshold_rules():
    y_val = np.r_[np.ones(5), np.zeros(95)].astype(int)
    p_val = np.r_[np.full(5, 0.8), np.full(95, 0.1)]
    y_te = np.r_[np.ones(10), np.zeros(1990)].astype(int)
    p_te = np.r_[np.full(10, 0.8), np.linspace(0, 0.2, 1990)]
    rows = {r["rule"]: r for r in C.threshold_report(y_val, p_val, y_te, p_te, pi_true=0.005)}
    assert rows["top 0.1% alert budget"]["n_flagged"] == 2     # 0.1% of 2,000 rows
    assert rows["tau = 0.5"]["recall"] == 1.0 and rows["tau = 0.5"]["precision"] == 1.0
    assert rows["tau = pi_1"]["n_flagged"] > rows["tau = 0.5"]["n_flagged"]   # lower bar, more alerts


def test_balanced_context_overpredicts_and_correction_fixes_it(tmp_path):
    # End to end on fake data: a balanced context makes the stand-in model think
    # fraud is common; after correction the mean prediction should be close to the
    # real fraud rate, and ECE should drop.
    rng = np.random.default_rng(1)
    y = (rng.random(60_000) < 0.004).astype(int)
    X = rng.normal(size=(60_000, 5)).astype(np.float32) + 1.0 * y[:, None]
    splits = make_splits(y, seeds=[0], levels={"natural": None})
    run_policy_sweep(X, y, splits, tmp_path, {"TabICLv2": lambda s: LogisticRegression(max_iter=300)},
                     ["uniform", "balanced"], levels=["natural"], budgets=[2_000], seeds=[0])
    run_xgb_sweep(X, y, splits, tmp_path, variants=["plain"], levels=["natural"], n_trials=2)
    calib, thresh = C.analyse_all(tmp_path, y)
    bal = calib[calib.policy == "balanced"].iloc[0]
    assert bal.mean_pred_raw > 3 * bal.observed_rate            # over-predicts before
    assert abs(bal.mean_pred_corrected - bal.observed_rate) < abs(bal.mean_pred_raw - bal.observed_rate)
    assert bal.ece_corrected < bal.ece_raw
    xgb = calib[calib.model == "XGBoost (plain)"].iloc[0]
    assert xgb.ece_raw == xgb.ece_corrected                     # no correction for full-data XGBoost
    assert set(thresh.rule) == {"tau = 0.5", "tau = pi_1", "tau = F1-best on validation", "top 0.1% alert budget"}
    pts = C.reliability_points(y[splits[0].test], np.full(len(splits[0].test), 0.01))
    C.plot_reliability({"constant": pts}, tmp_path / "f5.png", "test")
    assert (tmp_path / "f5.png").stat().st_size > 5_000
