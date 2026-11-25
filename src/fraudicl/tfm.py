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
#   one context for every row   : uniform, stratified, balanced, and the three that
#                                 choose the legitimate rows on purpose: cluster,
#                                 pca, hard_negative
#   one context per group of rows: knn, knn_balanced        (query-conditional)
#   several contexts, averaged   : ensemble_uniform, ensemble_balanced
# uniform_fresh is not a W1 policy: it is the confirmatory redraw of uniform.
SELECTED_POLICIES = ("cluster", "pca", "hard_negative")   # need the features, not only labels
# uniform_large (W4) is uniform with k above 10K (20K, 50K) for TabICLv2 only; same rows
# and order as uniform, kept under its own name so the 1K-10K tables stay as they are.
SINGLE_POLICIES = ("uniform", "stratified", "balanced", "uniform_fresh", "uniform_large") + SELECTED_POLICIES
GROUP_POLICIES = ("knn", "knn_balanced")
ENSEMBLE_POLICIES = ("ensemble_uniform", "ensemble_balanced")
POLICIES = SINGLE_POLICIES + GROUP_POLICIES + ENSEMBLE_POLICIES


def build_context(train_idx: np.ndarray, y: np.ndarray, k: int, policy: str, seed: int,
                  X: np.ndarray | None = None) -> np.ndarray:
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
    if policy in ("uniform", "uniform_large"):
        order = np.random.default_rng(seed).permutation(train_idx)
        return np.sort(order[:k])
    if policy == "uniform_fresh":
        # Confirmatory run (Nov 2026): the same rule as uniform, but a new random
        # order that no earlier run has used. The small-context finding was spotted
        # in the uniform runs, so it has to be checked on contexts I have not looked
        # at. Seeding with (seed, 2026) keeps this stream apart from uniform (seed),
        # the class orders (seed, 7) and the context-variance draws (1000 + i).
        order = np.random.default_rng([seed, 2026]).permutation(train_idx)
        return np.sort(order[:k])
    if policy in ("stratified", "balanced"):
        pos, neg = _class_orders(train_idx, y, seed)
        n_pos = _n_frauds(policy, k, len(pos), len(train_idx))
        return np.sort(np.concatenate([pos[:n_pos], neg[:k - n_pos]]))
    if policy in SELECTED_POLICIES:
        if X is None:
            raise ValueError(f"policy {policy!r} needs the feature matrix X")
        pos, neg = _class_orders(train_idx, y, seed)
        n_pos = _n_frauds("balanced", k, len(pos), len(train_idx))
        neg_order = _legit_order(policy, X, pos, neg, seed)
        return np.sort(np.concatenate([pos[:n_pos], neg_order[:k - n_pos]]))
    raise ValueError(f"unknown single-context policy {policy!r}")


# ---------------------------------------------------------------- choosing the legitimate rows
#
# The balanced policy showed that putting every fraud in the context is what helps
# most. These three keep exactly the same frauds as balanced and change only WHICH
# legitimate rows fill the rest, so any difference from balanced is caused by that
# choice. Each one returns the legitimate rows in an order; a context of size k takes
# a prefix of that order, so the contexts stay nested across k like all the others.
#
#   cluster        spread over 50 k-means clusters of the legitimate rows, taking one
#                  row from each cluster in turn. Rare kinds of legitimate transaction
#                  get represented, where a random draw mostly repeats the common ones.
#   pca            spread along the 3 leading principal components of the FRAUD rows:
#                  legitimate rows are binned by where they fall on those directions
#                  (5 x 5 x 5 cells) and taken from the cells in turn, so the model
#                  sees legitimate rows across the region where fraud varies.
#   hard_negative  half of the slots go to the legitimate rows closest to a fraud
#                  (the boundary cases the model most needs to separate), the other
#                  half to random legitimate rows, alternating. Pure hard negatives
#                  would show the model only fraud-like legitimate rows.
#
# Distances use features standardised with the training pool's mean and std, as in
# knn_groups, otherwise Amount and Time would decide everything. Only training rows
# are used, so validation and test rows can never influence a context.

N_CLUSTERS, N_PCS, N_BINS = 50, 3, 5


def _standardise(X, rows, ref):
    mu, sd = X[ref].mean(axis=0), X[ref].std(axis=0) + 1e-9
    return (X[rows] - mu) / sd


def _round_robin(groups, rng):
    """Interleave groups: one row from each group in turn, rows shuffled inside each
    group. A group that runs out simply drops out of the rotation."""
    queues = [rng.permutation(g) for g in groups if len(g)]
    out, i = [], 0
    while queues:
        queues = [q for q in queues if i < len(q)]
        out.extend(q[i] for q in queues)
        i += 1
    return np.asarray(out, dtype=np.int64)


def _legit_order(policy, X, pos, neg, seed):
    rng = np.random.default_rng([seed, 11])
    pool = np.concatenate([pos, neg])
    if policy == "cluster":
        from sklearn.cluster import MiniBatchKMeans
        Z = _standardise(X, neg, pool)
        labels = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=3,
                                 batch_size=4096).fit_predict(Z)
        return _round_robin([neg[labels == c] for c in range(N_CLUSTERS)], rng)
    if policy == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=N_PCS, random_state=seed).fit(_standardise(X, pos, pool))
        P = pca.transform(_standardise(X, neg, pool))
        # quantile bins on each component, so every bin holds a similar number of rows
        cells = np.zeros(len(neg), dtype=np.int64)
        for j in range(N_PCS):
            edges = np.quantile(P[:, j], np.linspace(0, 1, N_BINS + 1)[1:-1])
            cells = cells * N_BINS + np.searchsorted(edges, P[:, j])
        return _round_robin([neg[cells == c] for c in range(N_BINS ** N_PCS)], rng)
    if policy == "hard_negative":
        from sklearn.neighbors import NearestNeighbors
        dist, _ = NearestNeighbors(n_neighbors=1).fit(_standardise(X, pos, pool)).kneighbors(
            _standardise(X, neg, pool))
        hard = neg[np.argsort(dist[:, 0], kind="stable")]          # closest to a fraud first
        rand = rng.permutation(neg)
        out, seen = [], set()
        for h, r in zip(hard, rand):                               # alternate hard / random
            for row in (h, r):
                if row not in seen:
                    seen.add(row)
                    out.append(row)
        return np.asarray(out, dtype=np.int64)
    raise ValueError(policy)


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
    # Take the column of class 1 by name, not by position: a model fitted on a
    # context with only legitimate rows returns a single column, and [:, 1] would
    # crash. Such a model can only say "not fraud", so it scores 0 for every row.
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        return np.zeros(len(X))
    col = classes.index(1)
    out = [model.predict_proba(X[i:i + batch])[:, col] for i in range(0, len(X), batch)]
    return np.concatenate(out)
