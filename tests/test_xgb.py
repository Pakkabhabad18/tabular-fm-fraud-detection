"""Checks for the XGBoost arm (task T4) on a small fake dataset.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest

from fraudicl.xgb import VARIANTS, _resample_if_smote, fit_xgb, tune


@pytest.fixture(scope="module")
def data():
    # 4,000 rows, 5 features, 2% fraud; frauds are shifted so there is signal to learn.
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.02).astype(int)
    X = rng.normal(size=(4000, 5)).astype(np.float32) + 1.5 * y[:, None]
    return X[:3000], y[:3000], X[3000:], y[3000:]


@pytest.mark.parametrize("variant", VARIANTS)
def test_tune_runs_and_learns(data, variant):
    X_tr, y_tr, X_val, y_val = data
    best = tune(X_tr, y_tr, X_val, y_val, variant, n_trials=3)
    assert best["variant"] == variant
    # a model with real signal should beat the 2% a random guess scores
    assert best["val_pr_auc"] > 0.3
    if variant == "spw":
        assert "scale_pos_weight" in best["params"]
    if variant == "smote":
        assert "smote_ratio" in best["params"]


def test_smote_only_adds_frauds(data):
    X_tr, y_tr, *_ = data
    Xs, ys = _resample_if_smote(X_tr, y_tr, {"smote_ratio": 0.5}, seed=0)
    # legitimate rows unchanged in number, frauds raised to half the legitimate count
    assert (ys == 0).sum() == (y_tr == 0).sum()
    assert (ys == 1).sum() == int(0.5 * (y_tr == 0).sum())


def test_smote_skipped_when_already_balanced():
    X = np.zeros((10, 2)); y = np.array([1] * 5 + [0] * 5)
    Xs, ys = _resample_if_smote(X, y, {"smote_ratio": 0.25}, seed=0)
    assert len(ys) == 10


def test_fit_is_reproducible(data):
    X_tr, y_tr, X_val, y_val = data
    p = {"max_depth": 4, "learning_rate": 0.1}
    a = fit_xgb(X_tr, y_tr, X_val, y_val, p, seed=0).predict_proba(X_val)[:, 1]
    b = fit_xgb(X_tr, y_tr, X_val, y_val, p, seed=0).predict_proba(X_val)[:, 1]
    assert np.allclose(a, b)
