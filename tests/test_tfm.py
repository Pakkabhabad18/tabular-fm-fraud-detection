"""Checks for the context sampling and the T5/T6 loops, using a small logistic
regression in place of TabPFN / TabICL so the tests run anywhere without a GPU.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fraudicl.data import make_splits
from fraudicl.experiments import collect, get_context, run_matched_xgb, run_tfm_sweep, summarise_by_k
from fraudicl.tfm import build_context, feasible_budgets


@pytest.fixture(scope="module")
def fake():
    # 30,000 rows, 1% fraud, 6 features with some signal.
    rng = np.random.default_rng(0)
    y = (rng.random(30_000) < 0.01).astype(int)
    X = rng.normal(size=(30_000, 6)).astype(np.float32) + 1.2 * y[:, None]
    splits = make_splits(y, seeds=[0, 1], levels={"5": 0.05, "natural": None})
    return X, y, splits


def test_context_is_deterministic_and_inside_train(fake):
    X, y, splits = fake
    s = splits[0]
    a = build_context(s.train, y, 500, "uniform", seed=0)
    b = build_context(s.train, y, 500, "uniform", seed=0)
    assert np.array_equal(a, b)
    assert set(a) <= set(s.train)          # never uses validation or test rows
    assert len(np.unique(a)) == 500        # no row twice


def test_budget_larger_than_pool_is_skipped(fake):
    X, y, splits = fake
    s = next(s for s in splits if s.level == "5")
    assert len(s.train) < 10_000
    assert 10_000 not in feasible_budgets(s.train, [1_000, 2_000, 5_000, 10_000])
    with pytest.raises(ValueError):
        build_context(s.train, y, 10_000, "uniform", seed=0)


def test_tfm_sweep_and_matched_xgb_share_contexts(fake, tmp_path):
    X, y, splits = fake
    makers = {"Fake LR": lambda seed: LogisticRegression(max_iter=200)}
    run_tfm_sweep(X, y, splits, tmp_path, makers, levels=["5", "natural"],
                  budgets=[1_000, 5_000], seeds=[0, 1])
    run_matched_xgb(X, y, splits, tmp_path, levels=["5", "natural"],
                    budgets=[1_000, 5_000], seeds=[0, 1], n_trials=2)
    tfm = collect(tmp_path, "tfm")
    xgb = collect(tmp_path, "xgb_matched")
    # same cells, and each cell used the same context (same number of frauds)
    key = ["level", "k", "seed"]
    merged = tfm.merge(xgb, on=key, suffixes=("_tfm", "_xgb"))
    assert len(merged) == len(tfm) == len(xgb) > 0
    assert (merged.context_frauds_tfm == merged.context_frauds_xgb).all()
    assert (tfm.pr_auc > y.mean()).all()   # the stand-in model learns something
    # rerun skips everything already saved
    n_files = len(list((tmp_path / "raw" / "tfm").iterdir()))
    run_tfm_sweep(X, y, splits, tmp_path, makers, levels=["5", "natural"],
                  budgets=[1_000, 5_000], seeds=[0, 1])
    assert len(list((tmp_path / "raw" / "tfm").iterdir())) == n_files
    assert {"seeds_with_no_fraud", "context_frauds_mean"} <= set(summarise_by_k(tfm).columns)


def test_context_without_fraud_scores_as_no_skill(tmp_path):
    # 2 frauds in 20,000 training-like rows: a 100-row context almost surely has none.
    y = np.zeros(20_000, int); y[[5, 15_000]] = 1
    y[np.random.default_rng(1).choice(20_000, 60, replace=False)] = 1  # some frauds for val/test
    X = np.random.default_rng(2).normal(size=(20_000, 3)).astype(np.float32)
    splits = make_splits(y, seeds=[0], levels={"natural": None})
    s = splits[0]
    # find a budget whose context has no fraud
    for k in range(50, 400, 10):
        if y[build_context(s.train, y, k, "uniform", 0)].sum() == 0:
            break
    run_tfm_sweep(X, y, splits, tmp_path, {"Fake LR": lambda seed: LogisticRegression()},
                  levels=["natural"], budgets=[k], seeds=[0])
    row = collect(tmp_path, "tfm").iloc[0]
    assert row.context_frauds == 0
    # constant scores: PR-AUC equals the test fraud rate
    assert row.pr_auc == pytest.approx(row.fraud_rate)
