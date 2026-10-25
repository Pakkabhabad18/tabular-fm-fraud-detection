"""The modern arm: choosing context rows for the foundation models (task T5, then W1).

TabPFN v2 and TabICLv2 never train. "fit" only stores a table of labelled rows,
the context, and every prediction is made by attending to those rows. So the one
thing we control is which rows go into the context. That choice is a "context
policy". T5 uses the simplest one, uniform: k training rows drawn at random.
W1 (task T8) compares it with class-aware, query-conditional and ensemble policies.

Why uniform is the baseline worth beating: at the natural fraud rate a uniform
context of k rows holds about 0.0017 * k frauds on average, i.e. about 17 in 10K
rows and under 2 in 1K rows. Sometimes it holds none at all, and then the model
has no example of fraud to compare against.
"""

from __future__ import annotations

import numpy as np

# Context policies (W1). Three kinds, by how the context relates to the rows being predicted:
#   one context for every row   : uniform, stratified, balanced
#   one context per group of rows: knn, knn_balanced        (query-conditional)
#   several contexts, averaged   : ensemble_uniform, ensemble_balanced
SINGLE_POLICIES = ("uniform", "stratified", "balanced")
GROUP_POLICIES = ("knn", "knn_balanced")
ENSEMBLE_POLICIES = ("ensemble_uniform", "ensemble_balanced")
POLICIES = SINGLE_POLICIES + GROUP_POLICIES + ENSEMBLE_POLICIES


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
    if policy in ("stratified", "balanced"):
        pos, neg = _class_orders(train_idx, y, seed)
        n_pos = _n_frauds(policy, k, len(pos), len(train_idx))
        return np.sort(np.concatenate([pos[:n_pos], neg[:k - n_pos]]))
    raise ValueError(f"unknown single-context policy {policy!r}")


def _class_orders(train_idx, y, seed):
    """Frauds and legitimate rows of the pool, each in one fixed random order per seed.

    Taking prefixes of these orders keeps the stratified and balanced contexts
    nested across k, like the uniform ones. The generator is seeded with
    (seed, 7) so it does not reuse the uniform policy's random stream.
    """
    rng = np.random.default_rng([seed, 7])
    pos = rng.permutation(train_idx[y[train_idx] == 1])
    neg = rng.permutation(train_idx[y[train_idx] == 0])
    return pos, neg


def _n_frauds(policy: str, k: int, n_pos_pool: int, n_pool: int) -> int:
    """How many frauds a context of size k gets under each class-aware policy.

    stratified: the pool's own fraud rate, rounded, but at least 1. A uniform draw
                only matches the rate on average and can come out with zero frauds;
                stratified removes that luck. At the natural rate this is 2 frauds
                in 1K rows and 17 in 10K.
    balanced:   as many frauds as possible, up to half the context. The pool has
                only ~295 frauds, so below 1% fraud every context gets all of them
                (295 frauds in 1K rows = 29.5%, in 10K rows = 3%).
    """
    if policy == "stratified":
        return max(1, int(round(k * n_pos_pool / n_pool)))
    return min(n_pos_pool, k // 2)


def ensemble_contexts(train_idx, y, k, base: str, seed: int, m: int) -> list[np.ndarray]:
    """Up to m contexts of size k for the ensemble policies.

    ensemble_uniform:  the shuffled pool cut into consecutive, non-overlapping
                       chunks of k rows. The first chunk is exactly the uniform
                       context, so an ensemble of 1 equals the uniform policy.
    ensemble_balanced: every context holds the same frauds (as many as the
                       balanced policy uses) plus its own, non-overlapping slice of
                       legitimate rows. This is EasyEnsemble from classical
                       imbalanced learning, done with contexts instead of trees.
    Together the m contexts show the model m times more legitimate rows than one
    context can hold. Predictions are averaged in log-odds (see average_log_odds).
    """
    if base == "uniform":
        order = np.random.default_rng(seed).permutation(train_idx)
        n = min(m, len(order) // k)
        return [np.sort(order[i * k:(i + 1) * k]) for i in range(n)]
    if base == "balanced":
        pos, neg = _class_orders(train_idx, y, seed)
        n_pos = _n_frauds("balanced", k, len(pos), len(train_idx))
        chunk = k - n_pos
        n = min(m, len(neg) // chunk)
        return [np.sort(np.concatenate([pos[:n_pos], neg[i * chunk:(i + 1) * chunk]])) for i in range(n)]
    raise ValueError(f"unknown ensemble base {base!r}")


def knn_groups(train_idx, X, y, query_X, k, policy: str, seed: int, group_size: int = 1000):
    """Query-conditional contexts: one context per group of similar rows to predict.

    A separate context for every single test row would mean ~57,000 model fits.
    Instead the rows to predict are clustered (k-means, about group_size rows per
    cluster) and each cluster gets the k training rows nearest to its centre. All
    distances are in standardised features (mean and std of the training pool),
    otherwise Amount and Time, which have much larger ranges than V1-V28, would
    decide the neighbours on their own.

    knn:          the k nearest training rows, whatever their class. Near a
                  cluster of legitimate rows this can easily contain no fraud.
    knn_balanced: every fraud in the pool (up to k/2) plus the nearest legitimate
                  rows. The model always sees all fraud examples, and the
                  legitimate rows it compares against are the ones closest to the
                  rows it has to score.

    Yields (mask over query rows, context indices) for each group.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors

    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0) + 1e-9
    Q = (query_X - mu) / sd
    n_groups = max(1, int(np.ceil(len(Q) / group_size)))
    labels = MiniBatchKMeans(n_clusters=n_groups, random_state=seed, n_init=3,
                             batch_size=4096).fit_predict(Q)
    if policy == "knn":
        fixed = np.array([], dtype=train_idx.dtype)
        pool = train_idx
    elif policy == "knn_balanced":
        pos, neg = _class_orders(train_idx, y, seed)
        fixed = pos[:_n_frauds("balanced", k, len(pos), len(train_idx))]
        pool = neg
    else:
        raise ValueError(f"unknown group policy {policy!r}")
    n_take = k - len(fixed)
    nn = NearestNeighbors(n_neighbors=n_take).fit((X[pool] - mu) / sd)
    for g in range(n_groups):
        mask = labels == g
        if not mask.any():
            continue
        centre = Q[mask].mean(axis=0, keepdims=True)
        _, ind = nn.kneighbors(centre)
        yield mask, np.sort(np.concatenate([fixed, pool[ind[0]]]))


def average_log_odds(prob_list) -> np.ndarray:
    """Combine probabilities from several contexts by averaging their log-odds.

    Averaging log-odds instead of probabilities means a context that is very sure
    (p = 0.999) and one that is unsure (p = 0.5) combine like evidence adding up,
    and it keeps the ranking stable when contexts disagree in scale. This is the
    aggregation named in the proposal.
    """
    eps = 1e-7
    logits = [np.log(np.clip(p, eps, 1 - eps)) - np.log(np.clip(1 - p, eps, 1 - eps)) for p in prob_list]
    return 1 / (1 + np.exp(-np.mean(logits, axis=0)))


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
