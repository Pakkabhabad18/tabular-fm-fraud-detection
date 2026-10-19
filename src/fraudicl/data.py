"""Loading ULB and building the splits used by every experiment (task T2).

The splits are the single most important thing to get right: both model families
must be scored on exactly the same rows, or the comparison means nothing. So the
splits are built once here, from fixed seeds, and saved as row indices.

How a split for one (seed, prevalence level) is built:

1. Stratified 60/20/20 train / validation / test split of all 284,807 rows.
   Stratified means each part keeps the 0.172% fraud rate.
2. Inside each part, legitimate rows are put in a random order once per seed.
3. For a target fraud rate pi, every fraud in that part is kept, plus the first
   n_neg legitimate rows from that order, where n_neg = n_fraud * (1 - pi) / pi.

Because step 2 fixes one order per seed, the levels are nested: the rows at 20%
are a subset of the rows at 5%, which are a subset of the rows at 1%, and so on.
So when PR-AUC changes between two levels, it is because the fraud rate changed,
not because a different random sample of legitimate rows was drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PREVALENCE_LEVELS, SEEDS, SPLIT_FRACTIONS

ULB_ROWS = 284_807
ULB_FRAUDS = 492
TARGET = "Class"


def load_ulb() -> pd.DataFrame:
    """Download ULB with kagglehub (no token needed) and check it is the right file."""
    import kagglehub  # imported here so the rest of the module works without it

    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    check_ulb(df)
    return df


def check_ulb(df: pd.DataFrame) -> None:
    """Fail loudly if the data is not the ULB file described in the proposal."""
    assert df.shape == (ULB_ROWS, 31), f"expected 284,807 x 31, got {df.shape}"
    assert int(df[TARGET].sum()) == ULB_FRAUDS, "expected 492 frauds"
    assert df.isna().sum().sum() == 0, "ULB should have no missing values"
    assert list(df.columns[:2]) == ["Time", "V1"] and df.columns[-1] == TARGET


def features_and_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """All 30 columns (Time, V1-V28, Amount) as features, Class as the target.

    No scaling here: the foundation models normalise inputs themselves and trees
    don't need it. Any extra preprocessing belongs to a specific model.
    """
    X = df.drop(columns=TARGET).to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)
    return X, y


def n_negatives_for(n_pos: int, pi: float) -> int:
    """Number of legitimate rows that gives a fraud rate of pi with n_pos frauds."""
    return int(round(n_pos * (1 - pi) / pi))


@dataclass
class Split:
    """Row indices (into the full ULB table) for one seed and one fraud rate."""

    seed: int
    level: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def summary(self, y: np.ndarray) -> dict:
        """Row and fraud counts per part, used for the counts table and for checks."""
        row = {"seed": self.seed, "level": self.level}
        for part in ("train", "val", "test"):
            idx = getattr(self, part)
            row[f"{part}_rows"] = len(idx)
            row[f"{part}_frauds"] = int(y[idx].sum())
            row[f"{part}_fraud_rate"] = y[idx].mean()
        return row


def base_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60/20/20 split of all rows for one seed."""
    train_frac, val_frac, test_frac = SPLIT_FRACTIONS
    all_idx = np.arange(len(y))
    # First cut off the test part, then split the rest into train and validation.
    rest, test = train_test_split(
        all_idx, test_size=test_frac, stratify=y, random_state=seed
    )
    train, val = train_test_split(
        rest,
        test_size=val_frac / (train_frac + val_frac),  # 0.20 / 0.80 = 25% of the rest
        stratify=y[rest],
        random_state=seed,
    )
    return np.sort(train), np.sort(val), np.sort(test)


def _downsample(part: np.ndarray, y: np.ndarray, neg_order: np.ndarray, pi: float | None) -> np.ndarray:
    """Keep every fraud in `part` plus enough legitimate rows to reach fraud rate pi.

    neg_order is this part's legitimate rows in a fixed random order. Taking a prefix
    of it (rather than a fresh random draw) is what makes the levels nested.
    """
    pos = part[y[part] == 1]
    if pi is None:  # natural rate: keep the part as it is
        return part
    n_neg = n_negatives_for(len(pos), pi)
    if n_neg > len(neg_order):
        raise ValueError(f"not enough legitimate rows for pi={pi}")
    return np.sort(np.concatenate([pos, neg_order[:n_neg]]))


def make_splits(y: np.ndarray, seeds=SEEDS, levels=PREVALENCE_LEVELS) -> list[Split]:
    """Every (seed, level) split. Same inputs always give the same indices."""
    splits = []
    for seed in seeds:
        parts = base_split(y, seed)
        rng = np.random.default_rng(seed)
        # One random order of legitimate rows per part, fixed for this seed.
        neg_orders = [rng.permutation(p[y[p] == 0]) for p in parts]
        for level, pi in levels.items():
            train, val, test = (
                _downsample(p, y, order, pi) for p, order in zip(parts, neg_orders)
            )
            splits.append(Split(seed, level, train, val, test))
    return splits


def save_splits(splits: list[Split], path: str) -> None:
    """Save all split indices to one .npz file (kept on Drive, not in git)."""
    arrays = {}
    for s in splits:
        for part in ("train", "val", "test"):
            arrays[f"seed{s.seed}__{s.level}__{part}"] = getattr(s, part)
    np.savez_compressed(path, **arrays)


def load_splits(path: str) -> list[Split]:
    """Read back the file written by save_splits."""
    data = np.load(path)
    keys = sorted({k.rsplit("__", 1)[0] for k in data.files})
    out = []
    for key in keys:
        seed_tag, level = key.split("__")
        out.append(
            Split(
                seed=int(seed_tag.removeprefix("seed")),
                level=level,
                train=data[f"{key}__train"],
                val=data[f"{key}__val"],
                test=data[f"{key}__test"],
            )
        )
    return out


def fingerprint(splits: list[Split]) -> str:
    """A short hash of all indices. Committed to git so anyone can check that their
    rebuilt splits are byte-for-byte the ones used in the report."""
    import hashlib

    h = hashlib.sha256()
    for s in sorted(splits, key=lambda s: (s.seed, s.level)):
        for part in ("train", "val", "test"):
            h.update(np.ascontiguousarray(getattr(s, part), dtype=np.int64).tobytes())
    return h.hexdigest()[:16]
