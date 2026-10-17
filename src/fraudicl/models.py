"""Model factories.

Every notebook and script builds its models through these three functions, so the
model versions and default settings are decided in one place only.

Why this file exists: the tabpfn package (9.x) now loads TabPFN v3.5 by default.
v3.5 is a different, gated model and is not the one in the proposal. Calling
TabPFNClassifier() with no arguments would silently run the wrong model, so we
always ask for v2 explicitly here.
"""

from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import prepend_cache_path
from xgboost import XGBClassifier

# The exact checkpoints we use, pinned so a package update cannot swap the model.
#
# TabPFN v2: the package's own v2 default is "...-finetuned-zk73skhh.ckpt", a
# version fine-tuned on real datasets. We use the original checkpoint from the
# Nature paper (Hollmann et al., 2025) instead, because the proposal describes
# TabPFN as pretrained on synthetic data only and cites that paper.
TABPFN_V2_CHECKPOINT = "tabpfn-v2-classifier.ckpt"
TABICL_V2_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def make_tabpfn_v2(seed: int = 0, device: str = "auto") -> TabPFNClassifier:
    """TabPFN v2 with frozen weights (inference only).

    v2 was pretrained on tables of up to about 10K rows and 500 features. We leave
    ignore_pretraining_limits at its default (False), so the library raises an
    error if a context ever goes past those limits instead of quietly degrading.
    v2 weights are not gated, so no licence login is needed.
    """
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        model_path=prepend_cache_path(TABPFN_V2_CHECKPOINT),
        random_state=seed,
        device=device,
    )


def make_tabicl_v2(seed: int = 0, device: str | None = None) -> TabICLClassifier:
    """TabICLv2 with frozen weights (inference only). Open weights, no login needed."""
    return TabICLClassifier(
        checkpoint_version=TABICL_V2_CHECKPOINT,
        random_state=seed,
        device=device,
    )


def make_xgboost(seed: int = 0, **params) -> XGBClassifier:
    """XGBoost with sensible starting values.

    These defaults are only for the smoke test and quick checks. The real classical
    arm replaces them with the best of 50 Optuna trials (task T4). Any keyword
    passed in overrides the default of the same name.
    """
    settings = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",   # PR-AUC, our headline metric, also used for early stopping
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    settings.update(params)
    return XGBClassifier(**settings)
