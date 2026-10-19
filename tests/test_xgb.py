"""Checks for the XGBoost arm (task T4) on a small fake dataset.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest

import optuna

from fraudicl.experiments import archive_xgb_search_v1
from fraudicl.xgb import VARIANTS, _resample_if_smote, fit_xgb, suggest_params, tune


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


def test_search_space_allows_tiny_min_child_weight():
    # The tuner must be able to pick a min_child_weight far below 1, otherwise
    # a training set with 1-2 frauds can never be split.
    params = suggest_params(optuna.trial.FixedTrial({
        "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0,
        "min_child_weight": 1e-3, "gamma": 1e-8, "reg_alpha": 1e-8, "reg_lambda": 1e-8}), "plain", 10.0)
    assert params["min_child_weight"] == 1e-3


def test_two_frauds_learnable_with_small_min_child_weight():
    rng = np.random.default_rng(3)
    y = np.zeros(2000, int); y[[10, 900]] = 1
    X = rng.normal(size=(2000, 10)) + 2.0 * y[:, None]
    yv = (rng.random(5000) < 0.01).astype(int)
    Xv = rng.normal(size=(5000, 10)) + 2.0 * yv[:, None]
    from sklearn.metrics import average_precision_score as ap
    p = {"max_depth": 6, "learning_rate": 0.1, "min_child_weight": 0.1}
    assert ap(yv, fit_xgb(X, y, Xv, yv, p).predict_proba(Xv)[:, 1]) > 0.1
    p["min_child_weight"] = 1.0   # the old floor: no split possible, constant scores
    assert ap(yv, fit_xgb(X, y, Xv, yv, p).predict_proba(Xv)[:, 1]) < 0.05


def test_archive_xgb_search_moves_only_xgb(tmp_path):
    for rel in ["raw/xgb", "xgb_params", "raw/xgb_matched", "raw/tfm", "raw/contexts"]:
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "x.json").write_text("{}")
    assert archive_xgb_search_v1(tmp_path) is True
    assert not (tmp_path / "raw" / "xgb").exists() and not (tmp_path / "raw" / "xgb_matched").exists()
    assert (tmp_path / "raw" / "tfm" / "x.json").exists() and (tmp_path / "raw" / "contexts" / "x.json").exists()
    assert (tmp_path / "archive_v1_xgb_search" / "xgb_params" / "x.json").exists()
    assert archive_xgb_search_v1(tmp_path) is False
