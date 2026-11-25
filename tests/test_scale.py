"""Checks for the W4 scale and latency code, on fake data.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fraudicl import report as R
from fraudicl.data import make_splits
from fraudicl.experiments import run_tfm_sweep
from fraudicl.scale import latency_table, plot_scale, run_xgb_scale, scale_rows, scale_table
from fraudicl.tfm import build_context


@pytest.fixture(scope="module")
def fake():
    rng = np.random.default_rng(0)
    y = (rng.random(40_000) < 0.01).astype(int)
    X = rng.normal(size=(40_000, 5)).astype(np.float32) + 1.2 * y[:, None]
    return X, y, make_splits(y, seeds=[0, 1], levels={"natural": None})


def test_scale_rows_are_nested_and_match_uniform_context(fake):
    X, y, splits = fake
    s = splits[0]
    a, b = scale_rows(s.train, 1_000, 0), scale_rows(s.train, 5_000, 0)
    assert set(a) <= set(b) and len(b) == 5_000
    # the 1K subset is exactly the uniform 1K context, and uniform_large is uniform
    assert np.array_equal(a, build_context(s.train, y, 1_000, "uniform", 0))
    assert np.array_equal(build_context(s.train, y, 5_000, "uniform_large", 0),
                          build_context(s.train, y, 5_000, "uniform", 0))


def test_xgb_scale_curve_and_table(fake, tmp_path):
    X, y, splits = fake
    params = {"max_depth": 3, "learning_rate": 0.1}
    run_xgb_scale(X, y, splits, tmp_path, sizes=[1_000, 5_000, 10_000_000], seeds=[0, 1], params=params)
    run_tfm_sweep(X, y, splits, tmp_path, {"TabICLv2": lambda s: LogisticRegression(max_iter=300)},
                  levels=["natural"], budgets=[1_000], seeds=[0, 1])
    runs = R.load_runs(tmp_path)
    t = scale_table(runs)
    xgb = t[t.model == "XGBoost (plain, N rows)"]
    assert list(xgb.rows_seen) == [1_000, 5_000]          # sizes above the pool are skipped
    assert (xgb.n_seeds == 2).all() and (xgb.frauds_seen > 0).all()
    assert "TabICLv2" in set(t.model)
    plot_scale(t, tmp_path / "f7.png", full_xgb=0.8)
    assert (tmp_path / "f7.png").stat().st_size > 10_000
    # headline tables are untouched by the scale runs
    assert "XGBoost (plain, N rows)" not in set(R.uniform_only(runs).model)


def test_latency_table(fake):
    X, y, splits = fake
    m = LogisticRegression(max_iter=300).fit(X[:5_000], y[:5_000])
    t = latency_table({"logreg": m}, X, batch_sizes=(1, 100), repeats=3)
    assert list(t.batch_size) == [1, 100] and (t.seconds_per_call > 0).all()
    assert np.allclose(t.ms_per_transaction, 1000 * t.seconds_per_call / t.batch_size)
