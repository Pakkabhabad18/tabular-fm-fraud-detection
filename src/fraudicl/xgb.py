"""The classical arm: tuned XGBoost with three ways of handling imbalance (task T4).

The proposal's classical baseline has to be as strong as we can make it, so that
any win for the foundation models is believable. Three variants:

  plain  - ordinary fit; imbalance is handled only by choosing the decision
           threshold afterwards on validation data.
  spw    - scale_pos_weight: each fraud counts w times in the loss. w is tuned
           between 1 (no reweighting) and n_legit / n_fraud (full balance).
  smote  - SMOTE adds synthetic frauds to the TRAINING rows only, by interpolating
           between real frauds and their nearest fraud neighbours. Validation and
           test rows are never touched, otherwise we would be scoring on fake data.

Tuning: 50 Optuna TPE trials per (variant, fraud rate) on seed 0, each trial
early-stopped on validation PR-AUC. The best settings are then reused for seeds
1-4, refitting with early stopping on each seed's own validation set. The test set
is not used anywhere in tuning.
"""

from __future__ import annotations

import time

import numpy as np
import optuna
from xgboost import XGBClassifier

VARIANTS = ("plain", "spw", "smote")
SEARCH_VERSION = 2   # 2 = min_child_weight floor lowered from 1 to 1e-3 (20 Oct)
MAX_TREES = 2000          # upper limit; early stopping picks the real number
EARLY_STOP_ROUNDS = 50    # stop when validation PR-AUC hasn't improved for 50 trees


def _device() -> str:
    """Use the GPU when there is one (much faster on the full 170K-row training set)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def suggest_params(trial: optuna.Trial, variant: str, neg_pos_ratio: float) -> dict:
    """The search space: the proposal's depth, learning rate, subsampling, gamma,
    alpha and lambda, plus the variant's own imbalance setting."""
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # Floor 1e-3, not 1. min_child_weight is the least total hessian a leaf may hold;
        # at the start every row has hessian p(1-p), about 0.001 when frauds are 0.1% of
        # the rows. With a floor of 1 a leaf needs ~1,000 rows, so a training set with
        # 1-2 frauds cannot be split at all and XGBoost predicts a constant. The first
        # search (floor 1) made matched-budget XGBoost score exactly the fraud rate in
        # those cells, which was our search space failing, not XGBoost.
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if variant == "spw":
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight", 1.0, max(1.0, neg_pos_ratio), log=True
        )
    if variant == "smote":
        # target fraud:legit ratio after oversampling (1.0 = fully balanced)
        params["smote_ratio"] = trial.suggest_categorical(
            "smote_ratio", [0.05, 0.1, 0.25, 0.5, 1.0]
        )
    return params


def _resample_if_smote(X, y, params: dict, seed: int):
    """Apply SMOTE to the training rows if this variant asks for it."""
    ratio = params.get("smote_ratio")
    if ratio is None or y.mean() >= ratio / (1 + ratio):
        return X, y  # nothing to do, or the data is already at least that balanced
    from imblearn.over_sampling import SMOTE

    return SMOTE(sampling_strategy=ratio, k_neighbors=5, random_state=seed).fit_resample(X, y)


def fit_xgb(X_tr, y_tr, X_val, y_val, params: dict, seed: int = 0) -> XGBClassifier:
    """Fit one XGBoost model, early-stopped on validation PR-AUC."""
    X_fit, y_fit = _resample_if_smote(X_tr, y_tr, params, seed)
    xgb_params = {k: v for k, v in params.items() if k != "smote_ratio"}
    model = XGBClassifier(
        n_estimators=MAX_TREES,
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        eval_metric="aucpr",
        tree_method="hist",
        device=_device(),
        random_state=seed,
        n_jobs=-1,
        **xgb_params,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def tune(X_tr, y_tr, X_val, y_val, variant: str, n_trials: int = 50, seed: int = 0) -> dict:
    """Run the Optuna search for one variant and return the best settings.

    The score each trial is judged on is validation PR-AUC, the same metric we
    report on test, so the tuner optimises what we care about.
    """
    from sklearn.metrics import average_precision_score

    neg_pos_ratio = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    def objective(trial):
        params = suggest_params(trial, variant, neg_pos_ratio)
        model = fit_xgb(X_tr, y_tr, X_val, y_val, params, seed)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    return {
        "variant": variant,
        "params": study.best_params,
        "val_pr_auc": float(study.best_value),
        "best_iteration": study.best_trial.user_attrs["best_iteration"],
        "n_trials": n_trials,
        "tuning_seconds": round(time.perf_counter() - t0, 1),
    }
