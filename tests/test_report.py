"""Checks for the tables and figures (task T7), on a tiny fake sweep.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fraudicl import report as R
from fraudicl.data import make_splits
from fraudicl.experiments import run_matched_xgb, run_tfm_sweep, run_xgb_sweep


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    rng = np.random.default_rng(0)
    y = (rng.random(40_000) < 0.01).astype(int)
    X = rng.normal(size=(40_000, 5)).astype(np.float32) + 1.2 * y[:, None]
    levels = {"5": 0.05, "natural": None}
    splits = make_splits(y, seeds=[0, 1], levels=levels)
    # two stand-in "foundation models", so the code paths for both names run
    makers = {m: (lambda seed: LogisticRegression(max_iter=200)) for m in R.TFM_MODELS}
    run_tfm_sweep(X, y, splits, out, makers, levels=list(levels), budgets=[1_000, 2_000], seeds=[0, 1])
    run_matched_xgb(X, y, splits, out, levels=list(levels), budgets=[1_000, 2_000], seeds=[0, 1], n_trials=2)
    run_xgb_sweep(X, y, splits, out, variants=["plain"], levels=list(levels), n_trials=2)
    train_rows = {s.level: len(s.train) for s in splits if s.seed == 0}
    return out, y, R.load_runs(out), train_rows


def test_headline_uses_largest_k_and_all_models(sweep, monkeypatch):
    out, y, runs, train_rows = sweep
    monkeypatch.setattr(R, "CONTEXT_BUDGETS", [1_000, 2_000])
    h = R.headline_table(runs, train_rows)
    assert set(h.model) == {R.HEADLINE_XGB, *R.TFM_MODELS, R.MATCHED_XGB}
    assert (h[h.model != R.HEADLINE_XGB].k == 2_000).all()   # largest k that fits, never chosen by score
    assert (h.n_seeds == 2).all()


def test_paired_tests_cover_every_seed(sweep, monkeypatch):
    out, y, runs, train_rows = sweep
    monkeypatch.setattr(R, "CONTEXT_BUDGETS", [1_000, 2_000])
    p = R.paired_tests(out, y, runs, train_rows, n_boot=50)
    assert (p.n_seeds == 2).all()
    assert (p.seeds_tfm_worse + p.seeds_tfm_better + p.seeds_no_clear_difference == p.n_seeds).all()


def test_figures_are_written(sweep, tmp_path, monkeypatch):
    out, y, runs, train_rows = sweep
    monkeypatch.setattr(R, "CONTEXT_BUDGETS", [1_000, 2_000])
    R.plot_headline(R.headline_table(runs, train_rows), tmp_path / "f1.png")
    R.plot_by_k(R.by_k_table(runs), runs, tmp_path / "f2.png")
    R.plot_vs_context_frauds(runs, tmp_path / "f3.png")
    assert all((tmp_path / f).stat().st_size > 10_000 for f in ("f1.png", "f2.png", "f3.png"))


def test_headline_has_f1_and_recall_at_fpr(sweep, monkeypatch):
    out, y, runs, train_rows = sweep
    monkeypatch.setattr(R, "CONTEXT_BUDGETS", [1_000, 2_000])
    h = R.headline_table(runs, train_rows)
    for col in ("f1_mean", "f1_std", "recall_at_fpr_0.1pct_mean"):
        assert col in h and h[col].between(0, 1).all()
    assert "f1_mean" in R.by_k_table(runs)


def test_rescore_changes_only_the_test_set(sweep, tmp_path, monkeypatch):
    out, y, runs, train_rows = sweep
    monkeypatch.setattr(R, "CONTEXT_BUDGETS", [1_000, 2_000])
    # the fixture's splits, rebuilt the same way (make_splits is deterministic)
    splits = make_splits(y, seeds=[0, 1], levels={"5": 0.05, "natural": None})
    c = R.rescore_by_test_mix(out, y, splits, train_rows)
    assert set(c.model) == {R.HEADLINE_XGB, *R.TFM_MODELS, R.MATCHED_XGB}
    assert set(c.test_level) == {"5", "natural"}
    # scored on its own test set, the control must give back the saved PR-AUC
    nat = c[c.test_level == "natural"].merge(
        runs[runs.level == "natural"].assign(k=runs.k.where(runs.model != R.HEADLINE_XGB)),
        on=["model", "seed"], suffixes=("", "_saved"))
    nat = nat[nat.k.isna() | (nat.k == nat.k_saved)]
    assert np.allclose(nat.pr_auc, nat.pr_auc_saved)
    # same frauds and same scores on every test mix: recall at the fixed threshold
    # cannot move, only precision can
    spread = c.groupby(["model", "seed"]).recall_at_val_threshold.agg(lambda r: r.max() - r.min())
    assert (spread < 1e-12).all()
    assert (c.groupby(["model", "seed"]).test_frauds.nunique() == 1).all()
    s = R.rescore_summary(c, R.headline_table(runs, train_rows))
    assert s.pr_auc_own_model.notna().all()
    R.plot_rescore_by_test_mix(s, tmp_path / "f6.png")
    assert (tmp_path / "f6.png").stat().st_size > 10_000
