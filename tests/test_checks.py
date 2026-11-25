"""Checks for the leakage / memorisation tests in checks.py.

Run with:  python -m pytest tests -q
A logistic regression stands in for the foundation models so this runs on a CPU.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from fraudicl import checks as C
from fraudicl.data import make_splits
from fraudicl.experiments import run_matched_xgb, run_tfm_sweep
from fraudicl.report import load_runs
from fraudicl.tfm import ensemble_contexts, knn_groups

LR = {"TabICLv2": lambda s: LogisticRegression(max_iter=300)}


def _fake(n=40_000, rate=0.01, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = rng.normal(size=(n, 5)).astype(np.float32) + 1.5 * y[:, None]
    return X, y


def test_saved_contexts_stay_inside_training_rows(tmp_path):
    X, y = _fake()
    splits = make_splits(y, seeds=[0, 1], levels={"natural": None})
    run_tfm_sweep(X, y, splits, tmp_path, LR, levels=["natural"], budgets=[1_000, 2_000],
                  seeds=[0, 1], policy="balanced")
    leak = C.context_leakage(tmp_path, splits)
    assert len(leak) == 4
    assert (leak[["outside_train", "in_validation", "in_test"]] == 0).all().all()
    # A context that contains a test row must be caught.
    s = splits[0]
    np.save(tmp_path / "raw" / "contexts" / "uniform__natural__k3__seed0.npy",
            np.r_[s.train[:2], s.test[:1]])
    leak = C.context_leakage(tmp_path, splits)
    assert leak[leak.k == 3].in_test.iloc[0] == 1


def test_ensemble_and_knn_contexts_come_from_training_rows():
    X, y = _fake()
    s = make_splits(y, seeds=[0], levels={"natural": None})[0]
    for ctx in ensemble_contexts(s.train, y, 1_000, "balanced", 0, 5):
        assert np.isin(ctx, s.train).all()
    for _, ctx in knn_groups(s.train, X, y, X[s.test], 1_000, "knn_balanced", 0):
        assert np.isin(ctx, s.train).all()


def test_duplicates_are_found_and_removed_from_scoring(tmp_path):
    X, y = _fake()
    splits = make_splits(y, seeds=[0], levels={"natural": None})
    s = splits[0]
    # Copy 3 training frauds onto 3 test frauds: same features, different row index.
    tr_f = s.train[y[s.train] == 1][:3]
    te_f = s.test[y[s.test] == 1][:3]
    X[te_f] = X[tr_f]
    dup = C.duplicate_overlap(X, y, splits).iloc[0]
    assert dup.test_frauds_with_copy_in_train == 3
    run_tfm_sweep(X, y, splits, tmp_path, LR, levels=["natural"], budgets=[1_000],
                  seeds=[0], policy="uniform")
    res = C.rescore_without_duplicates(tmp_path, X, y, splits, families=("tfm",))
    assert res.frauds_dropped.iloc[0] == 3
    assert np.isfinite(res.pr_auc_without_duplicates).all()


def test_shuffled_labels_fall_to_chance():
    X, y = _fake(rate=0.02)
    s = make_splits(y, seeds=[0], levels={"natural": None})[0]
    ctx = s.train[:5_000]
    res = C.shuffled_label_control(X, y, s, ctx, LR["TabICLv2"], "stand-in").set_index("context_labels")
    assert res.loc["real", "pr_auc"] > 0.5
    # with shuffled labels the model can only guess: close to the fraud rate
    assert res.loc["shuffled", "pr_auc"] < 3 * res.loc["shuffled", "no_skill_pr_auc"] + 0.05


def test_fast_average_precision_matches_sklearn():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(3)
    y = (rng.random(3_000) < 0.05).astype(int)
    s = np.round(rng.random(3_000) + 0.3 * y, 2)          # rounded, so there are ties
    ap = C._SortedAP(y, s)
    assert np.isclose(ap(np.ones(len(y))), average_precision_score(y, s))
    # integer weights = a bootstrap resample; compare with sklearn on repeated rows
    w = rng.integers(0, 3, len(y)).astype(float)
    w[np.flatnonzero(y)[0]] = 1                            # keep at least one fraud
    rep = np.repeat(np.arange(len(y)), w.astype(int))
    assert np.isclose(ap(w[ap.order]), average_precision_score(y[rep], s[rep]))


def test_holm_correction():
    adj = C.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_matched_tests_table(tmp_path):
    X, y = _fake()
    splits = make_splits(y, seeds=[0, 1], levels={"natural": None})
    run_tfm_sweep(X, y, splits, tmp_path, LR, levels=["natural"], budgets=[1_000, 2_000],
                  seeds=[0, 1], policy="uniform")
    run_matched_xgb(X, y, splits, tmp_path, levels=["natural"], budgets=[1_000, 2_000],
                    seeds=[0, 1], policy="uniform", variant="plain", n_trials=2)
    t = C.matched_tests(tmp_path, y, load_runs(tmp_path), models=("TabICLv2",), n_boot=200)
    assert len(t) == 2 and (t.n_seeds == 2).all()
    assert (t.p_holm >= t.p_value - 1e-12).all()
    assert (t.ci_low <= t.mean_diff).all() and (t.mean_diff <= t.ci_high).all()


def test_confirmatory_verdict_follows_the_fixed_rule():
    import pandas as pd
    from fraudicl.checks import confirmatory_verdict
    levels = ["20", "5", "5", "1", "1", "0.5", "0.5", "natural", "natural"]
    ks = [1_000, 1_000, 2_000, 1_000, 2_000, 1_000, 2_000, 1_000, 2_000]
    def table(diffs):
        return pd.DataFrame({"model": ["A"] * 9 + ["B"] * 9, "level": levels * 2, "k": ks * 2,
                             "mean_diff": diffs, "p_holm": [0.5] * 18})
    yes = confirmatory_verdict(table([0.01] * 7 + [0.05, 0.05] + [0.01] * 4 + [-0.01] * 3 + [0.04, 0.04]))
    assert yes["cells_tfm_ahead"] == 15 and yes["part1_direction_confirmed"] and yes["part2_largest_at_natural"]
    no = confirmatory_verdict(table([0.01] * 6 + [-0.01] * 12))
    assert not no["part1_direction_confirmed"]   # 6 of 18 is not a majority


def test_knn_diagnosis_on_a_small_run(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from fraudicl.checks import knn_diagnosis
    from fraudicl.data import make_splits
    from fraudicl.experiments import run_policy_sweep
    rng = np.random.default_rng(0)
    y = (rng.random(40_000) < 0.005).astype(int)
    X = rng.normal(size=(40_000, 5)).astype(np.float32) + 1.2 * y[:, None]
    splits = make_splits(y, seeds=[0], levels={"natural": None})
    run_policy_sweep(X, y, splits, tmp_path, {"TabICLv2": lambda s: LogisticRegression(max_iter=300)},
                     ["knn", "knn_balanced"], levels=["natural"], budgets=[1_000], seeds=[0])
    for policy in ("knn", "knn_balanced"):
        d = knn_diagnosis(tmp_path, X, y, splits[0], policy, 1_000)
        assert d["n_groups"] >= 1 and 0 <= d["share_test_rows_no_fraud_context"] <= 1
        assert 0 <= d["pooled_pr_auc"] <= 1 and 1 <= d["top_100_groups"] <= d["n_groups"]
    # knn_balanced always puts every fraud in the context, so no group goes without
    assert knn_diagnosis(tmp_path, X, y, splits[0], "knn_balanced", 1_000)["groups_without_fraud"] == 0
