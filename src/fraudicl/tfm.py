"""The modern arm: choosing context rows for the foundation models (task T5, then W1).

TabPFN v2 and TabICLv2 never train. "fit" only stores a table of labelled rows,
the context, and every prediction is made by attending to those rows. So the one
thing we control is which rows go into the context. That choice is a "context
policy". T5 uses the simplest one, uniform: k training rows drawn at random.
W1 adds the class-aware policies to this file later.

Why uniform is the baseline worth beating: at the natural fraud rate a uniform
context of k rows holds about 0.0017 * k frauds on average, i.e. about 17 in 10K
rows and under 2 in 1K rows. Sometimes it holds none at all, and then the model
has no example of fraud to compare against.
"""

from __future__ import annotations

import numpy as np

POLICIES = ("uniform",)


def build_context(train_idx: np.ndarray, y: np.ndarray, k: int, policy: str, seed: int) -> np.ndarray:
    """Pick k context rows from one split's training rows.

    Uniform contexts are nested across k, the same way the fraud-rate levels are
    nested: each seed puts the training pool in one fixed random order, and a
    budget of k takes the first k rows of that order. So the 2K context is the 1K
    context plus 1K more rows, and a change in PR-AUC from k = 1K to 2K comes from
    those extra rows, not from a fresh random draw. (A first version drew each k
    independently; on seed 0 that made the k-curves jump around, e.g. 0.5% fraud
    with k = 2K holding fewer frauds than k = 1K.)

    The order depends only on the seed, so the matched-budget XGBoost (T6) can be
    trained on exactly the rows the foundation model saw.
    """
    if k > len(train_idx):
        raise ValueError(f"k={k} is larger than the training pool ({len(train_idx)} rows)")
    if policy == "uniform":
        order = np.random.default_rng(seed).permutation(train_idx)
        return np.sort(order[:k])
    raise ValueError(f"unknown policy {policy!r}")


def feasible_budgets(train_idx: np.ndarray, budgets) -> list[int]:
    """Budgets that fit in this training pool. At a 20% fraud rate the pool has only
    about 1,475 rows, so only k = 1K is possible there."""
    return [k for k in budgets if k <= len(train_idx)]


def predict_in_batches(model, X: np.ndarray, batch: int = 10_000) -> np.ndarray:
    """Fraud probability for every row of X, predicted in chunks.

    Prediction memory grows with (context rows x rows predicted at once), so
    chunking keeps a 57K-row validation or test set within the T4's 15 GB.
    """
    out = [model.predict_proba(X[i:i + batch])[:, 1] for i in range(0, len(X), batch)]
    return np.concatenate(out)
