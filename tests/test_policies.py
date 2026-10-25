"""Checks for the W1 context policies (task T8), on fake data with a logistic
regression standing in for the foundation models.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fraudicl import report as R
from fraudicl.data import make_splits
from fraudicl.experiments import collect, run_context_variance, run_policy_sweep
from fraudicl.tfm import (POLICIES, average_log_odds, build_context, ensemble_contexts,
                          knn_groups)


@pytest.fixture(scope="module")
def fake():
    # 60,000 rows at 0.5% fraud, so the natural pool looks like ULB's: few frauds, many legit
    rng = np.random.default_rng(0)
    y = (rng.random(60_000) < 0.005).astype(int)
    X = rng.normal(size=(60_000, 6)).astype(np.float32) + 1.2 * y[:, None]
    X[:, 0] *= 1000   # one feature on a much larger scale, like Amount
    splits = make_splits(y, seeds=[0, 1], levels={"1": 0.01, "natural": None})
    return X, y, splits


def _pool(splits, level="natural", seed=0):
    return next(s for s in splits if s.level == level and s.seed == seed)


def test_uniform_unchanged_from_t5(fake):
    # the T5 runs on Drive were made with this exact rule; W1 must not change it
    X, y, splits = fake
    s = _pool(splits)
    old = np.sort(np.random.default_rng(0).permutation(s.train)[:2_000])
    assert np.array_equal(build_context(s.train, y, 2_000, "uniform", 0), old)


def test_stratified_keeps_rate_and_at_least_one_fraud(fake):
    X, y, splits = fake
    s = _pool(splits)
    rate = y[s.train].mean()
    for k in (100, 1_000, 5_000):
        ctx = build_context(s.train, y, k, "stratified", 0)
        assert len(ctx) == k and set(ctx) <= set(s.train)
        assert y[ctx].sum() == max(1, round(k * rate))


def test_balanced_takes_all_frauds_up_to_half(fake):
    X, y, splits = fake
    s = _pool(splits)
    n_pos = int(y[s.train].sum())
    for k in (100, 1_000, 5_000):
        ctx = build_context(s.train, y, k, "balanced", 0)
        assert len(ctx) == k and y[ctx].sum() == min(n_pos, k // 2)


@pytest.mark.parametrize("policy", ["uniform", "stratified", "balanced"])
def test_class_aware_contexts_are_nested(fake, policy):
    X, y, splits = fake
    s = _pool(splits)
    small, big = (set(build_context(s.train, y, k, policy, 0)) for k in (1_000, 5_000))
    assert small <= big


def test_ensemble_uniform_disjoint_and_starts_with_uniform(fake):
    X, y, splits = fake
    s = _pool(splits)
    ctxs = ensemble_contexts(s.train, y, 2_000, "uniform", 0, m=5)
    assert len(ctxs) == 5
    assert np.array_equal(ctxs[0], build_context(s.train, y, 2_000, "uniform", 0))
    all_rows = np.concatenate(ctxs)
    assert len(np.unique(all_rows)) == len(all_rows)   # no row in two contexts


def test_ensemble_balanced_shares_frauds_not_legit(fake):
    X, y, splits = fake
    s = _pool(splits)
    ctxs = ensemble_contexts(s.train, y, 2_000, "balanced", 0, m=4)
    frauds = [set(c[y[c] == 1]) for c in ctxs]
    legit = np.concatenate([c[y[c] == 0] for c in ctxs])
    assert all(f == frauds[0] for f in frauds)
    assert len(np.unique(legit)) == len(legit)
    assert np.array_equal(ctxs[0], build_context(s.train, y, 2_000, "balanced", 0))


@pytest.mark.parametrize("policy", ["knn", "knn_balanced"])
def test_knn_groups_cover_every_query_row_once(fake, policy):
    X, y, splits = fake
    s = _pool(splits)
    seen = np.zeros(len(s.test), int)
    n_pos = int(y[s.train].sum())
    for mask, ctx in knn_groups(s.train, X, y, X[s.test], 1_000, policy, seed=0):
        seen += mask
        assert len(ctx) == 1_000 and set(ctx) <= set(s.train)
        if policy == "knn_balanced":
            assert y[ctx].sum() == min(n_pos, 500)
    assert (seen == 1).all()


def test_knn_neighbours_use_standardised_distance(fake):
    # with feature 0 scaled by 1000, raw distances would pick neighbours by feature 0 only;
    # after standardising, the retrieved rows should be close on the other features too
    X, y, splits = fake
    s = _pool(splits)
    q = s.test[:300]
    for mask, ctx in knn_groups(s.train, X, y, X[q], 500, "knn", seed=0):
        centre = X[q][mask][:, 1:].mean(0)
        near = np.abs(X[ctx][:, 1:].mean(0) - centre).mean()
        far = np.abs(X[s.train][:, 1:].mean(0) - centre).mean()
        assert near <= far + 1e-6


def test_average_log_odds():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(average_log_odds([p, p]), p)
    assert np.allclose(average_log_odds([np.full(3, 0.2), np.full(3, 0.8)]), 0.5)


def test_policy_sweep_and_table(fake, tmp_path):
    X, y, splits = fake
    makers = {"TabICLv2": lambda seed: LogisticRegression(max_iter=300)}
    run_policy_sweep(X, y, splits, tmp_path, makers, POLICIES, levels=["1", "natural"],
                     budgets=[1_000, 2_000], seeds=[0, 1], m_ensemble=3)
    runs = R.load_runs(tmp_path)
    assert set(runs.policy) == set(POLICIES)
    t = R.policy_table(runs, "TabICLv2")
    uni = t[t.policy == "uniform"]
    assert (uni.diff_vs_uniform_mean == 0).all()   # uniform against itself
    assert (t.n_seeds == 2).all()
    # every run learned something
    assert (runs.pr_auc > runs.fraud_rate).all()
    # rerun skips everything already saved
    n = len(collect(tmp_path, "tfm"))
    run_policy_sweep(X, y, splits, tmp_path, makers, POLICIES, levels=["1", "natural"],
                     budgets=[1_000, 2_000], seeds=[0, 1], m_ensemble=3)
    assert len(collect(tmp_path, "tfm")) == n
    R.plot_policies(t, runs, tmp_path / "f4.png")
    assert (tmp_path / "f4.png").stat().st_size > 10_000


def test_context_variance_fixed_split(fake, tmp_path):
    X, y, splits = fake
    s = _pool(splits)
    make = lambda seed: LogisticRegression(max_iter=300)
    a = run_context_variance(X, y, s, tmp_path, make, "TabICLv2", 2_000, context_seeds=range(4))
    assert len(a) == 4 and a.split_seed.nunique() == 1
    assert a.context_frauds.nunique() > 1     # different draws, different contexts
    b = run_context_variance(X, y, s, tmp_path, make, "TabICLv2", 2_000, context_seeds=range(4))
    assert np.allclose(a.pr_auc, b.pr_auc)    # rerun reads the saved results
