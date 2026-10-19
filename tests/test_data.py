"""Checks for the split code (task T2), run on fake labels shaped like ULB.

Run with:  python -m pytest tests -q
"""
import numpy as np
import pytest

from fraudicl.config import PREVALENCE_LEVELS, SEEDS
from fraudicl.data import ULB_FRAUDS, ULB_ROWS, fingerprint, load_splits, make_splits, save_splits


@pytest.fixture(scope="module")
def y():
    # Same size and fraud count as ULB, frauds at random positions.
    rng = np.random.default_rng(123)
    y = np.zeros(ULB_ROWS, dtype=np.int64)
    y[rng.choice(ULB_ROWS, ULB_FRAUDS, replace=False)] = 1
    return y


@pytest.fixture(scope="module")
def splits(y):
    return make_splits(y)


def test_one_split_per_seed_and_level(splits):
    assert len(splits) == len(SEEDS) * len(PREVALENCE_LEVELS)


def test_parts_do_not_overlap(splits):
    # A row used for training must never appear in validation or test.
    for s in splits:
        assert not set(s.train) & set(s.val)
        assert not set(s.train) & set(s.test)
        assert not set(s.val) & set(s.test)


def test_natural_level_uses_every_row(splits, y):
    for s in splits:
        if s.level == "natural":
            assert len(s.train) + len(s.val) + len(s.test) == ULB_ROWS


def test_every_fraud_kept_at_every_level(splits, y):
    # Downsampling only ever removes legitimate rows.
    for s in splits:
        n_fraud = sum(int(y[getattr(s, p)].sum()) for p in ("train", "val", "test"))
        assert n_fraud == ULB_FRAUDS


def test_fraud_rate_hits_target(splits, y):
    for s in splits:
        pi = PREVALENCE_LEVELS[s.level]
        for part in ("train", "val", "test"):
            rate = y[getattr(s, part)].mean()
            target = pi if pi is not None else ULB_FRAUDS / ULB_ROWS
            # rounding n_neg to a whole row moves the rate by a tiny amount only
            assert abs(rate - target) / target < 0.01


def test_levels_are_nested(splits):
    # Rows at a higher fraud rate are a subset of rows at the next lower rate.
    order = list(PREVALENCE_LEVELS)  # 20, 5, 1, 0.5, natural
    by_key = {(s.seed, s.level): s for s in splits}
    for seed in SEEDS:
        for hi, lo in zip(order, order[1:]):
            for part in ("train", "val", "test"):
                assert set(getattr(by_key[seed, hi], part)) <= set(getattr(by_key[seed, lo], part))


def test_same_seed_same_indices(y, splits):
    again = make_splits(y)
    assert fingerprint(again) == fingerprint(splits)


def test_seeds_differ(splits):
    a = next(s for s in splits if s.seed == 0 and s.level == "natural")
    b = next(s for s in splits if s.seed == 1 and s.level == "natural")
    assert not np.array_equal(a.test, b.test)


def test_save_load_roundtrip(tmp_path, splits):
    path = tmp_path / "splits.npz"
    save_splits(splits, path)
    assert fingerprint(load_splits(path)) == fingerprint(splits)
