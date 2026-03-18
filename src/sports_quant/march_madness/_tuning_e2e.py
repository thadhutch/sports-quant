"""End-to-end Optuna tuning through the stacking meta-learner pipeline.

Phase 1: Re-tune LightGBM hyperparameters, evaluating via meta-learner
         log loss on held-out folds (not raw LightGBM log loss).
Phase 2: Tune meta-learner params (LR, RF, meta-LR) with frozen LightGBM.

Both phases use the same temporal CV folds from model_config.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import log_loss

from sports_quant import _config as config
from sports_quant.march_madness._config import load_mm_config
from sports_quant.march_madness._data import (
    compute_combined_difference_features,
    compute_difference_features,
    symmetrize_training_data,
)
from sports_quant.march_madness._features import TARGET_COLUMN, YEAR_COLUMN
from sports_quant.march_madness._meta_learner import (
    _build_lr,
    _build_rf,
    _train_lgbm_ensemble,
    _train_sklearn_predict,
    collect_stacked_oof,
    train_meta_learner,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature preparation (mirrors backtest._prepare_features)
# ---------------------------------------------------------------------------


def _prepare_features(df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    """Extract feature matrix from a raw matchup DataFrame."""
    if feature_mode == "combined":
        return compute_combined_difference_features(df)
    if feature_mode == "difference":
        return compute_difference_features(df)

    from sports_quant.march_madness._features import DROP_COLUMNS

    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    filtered = df.drop(columns=drop_cols)
    return filtered.drop(columns=[TARGET_COLUMN], errors="ignore")


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------


def _evaluate_meta_pipeline_on_fold(
    matchups_df: pd.DataFrame,
    train_end: int,
    val_year: int,
    lgbm_params: dict,
    meta_cfg: dict,
    feature_mode: str,
    do_symmetrize: bool,
    early_stop_rounds: int,
) -> float:
    """Evaluate the full stacking pipeline on one temporal CV fold.

    1. Collect stacked OOF from years ≤ train_end (inner temporal CV).
    2. Train meta-learner on those OOF predictions.
    3. Train all 3 base learners on years ≤ train_end, predict on val_year.
    4. Stack predictions → meta-learner predicts → return log_loss.

    Returns:
        Log loss of the meta-learner predictions on val_year.
    """
    if train_end >= val_year:
        raise ValueError(
            f"Data leakage: train_end ({train_end}) >= val_year "
            f"({val_year}). train_end must be strictly less than "
            f"val_year."
        )

    available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())
    prior_years = [y for y in available_years if y <= train_end]

    # Step 1: Collect OOF predictions via inner temporal CV
    oof = collect_stacked_oof(
        matchups_df=matchups_df,
        prior_years=prior_years,
        prepare_features_fn=_prepare_features,
        hyperparams=lgbm_params,
        meta_cfg=meta_cfg,
        do_symmetrize=do_symmetrize,
        feature_mode=feature_mode,
        early_stop_rounds=early_stop_rounds,
    )

    # Step 2: Train meta-learner on OOF
    meta_model = train_meta_learner(oof, meta_cfg)

    # Step 3: Train base learners on all prior data, predict on val_year
    all_prior_df = matchups_df[matchups_df[YEAR_COLUMN].isin(prior_years)]
    val_df = matchups_df[matchups_df[YEAR_COLUMN] == val_year]

    if len(val_df) == 0:
        return float("inf")

    X_train = _prepare_features(all_prior_df, feature_mode)
    y_train = all_prior_df[TARGET_COLUMN].reset_index(drop=True)
    X_val = _prepare_features(val_df, feature_mode)
    y_val = val_df[TARGET_COLUMN].reset_index(drop=True)

    if do_symmetrize and feature_mode in ("difference", "combined"):
        X_train_s, y_train_s = symmetrize_training_data(X_train, y_train)
    else:
        X_train_s, y_train_s = X_train, y_train

    n_lgbm = meta_cfg.get("lgbm_oof_ensemble_size", 10)
    lgbm_preds = _train_lgbm_ensemble(
        X_train_s, y_train_s, X_val, lgbm_params,
        n_models=n_lgbm,
        early_stop_rounds=early_stop_rounds,
    )
    lr_preds = _train_sklearn_predict(
        _build_lr(meta_cfg), X_train_s, y_train_s, X_val,
    )
    rf_preds = _train_sklearn_predict(
        _build_rf(meta_cfg), X_train_s, y_train_s, X_val,
    )

    # Step 4: Stack and predict
    base_preds = np.column_stack([lgbm_preds, lr_preds, rf_preds])
    meta_preds = meta_model.predict_proba(base_preds)[:, 1]

    return float(log_loss(y_val, meta_preds))


# ---------------------------------------------------------------------------
# Full CV evaluation (all folds)
# ---------------------------------------------------------------------------


def evaluate_e2e_on_folds(
    matchups_df: pd.DataFrame,
    cv_folds: list[dict],
    lgbm_params: dict,
    meta_cfg: dict,
    feature_mode: str,
    do_symmetrize: bool,
    early_stop_rounds: int,
) -> dict:
    """Evaluate the full meta-learner pipeline across all CV folds.

    Returns:
        Dict with per-fold log losses and aggregate stats.
    """
    fold_results = []

    for fold in cv_folds:
        train_end = fold["train_end"]
        val_year = fold["val_year"]

        available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())
        if val_year not in available_years:
            continue

        fold_loss = _evaluate_meta_pipeline_on_fold(
            matchups_df=matchups_df,
            train_end=train_end,
            val_year=val_year,
            lgbm_params=lgbm_params,
            meta_cfg=meta_cfg,
            feature_mode=feature_mode,
            do_symmetrize=do_symmetrize,
            early_stop_rounds=early_stop_rounds,
        )

        fold_results.append({
            "fold": f"train≤{train_end}→val{val_year}",
            "train_end": train_end,
            "val_year": val_year,
            "meta_log_loss": fold_loss,
        })

        logger.info(
            "Fold train≤%d→val%d: meta log_loss=%.4f",
            train_end, val_year, fold_loss,
        )

    if not fold_results:
        raise ValueError(
            "No CV folds could be evaluated — val_year not found in data"
        )

    mean_loss = float(np.mean([r["meta_log_loss"] for r in fold_results]))
    std_loss = float(np.std([r["meta_log_loss"] for r in fold_results]))

    return {
        "folds": fold_results,
        "mean_meta_log_loss": mean_loss,
        "std_meta_log_loss": std_loss,
    }


# ---------------------------------------------------------------------------
# Phase 1: Tune LightGBM through meta-learner
# ---------------------------------------------------------------------------


def run_phase1_lgbm_tuning(
    matchups_df: pd.DataFrame,
    cv_folds: list[dict],
    meta_cfg: dict,
    feature_mode: str,
    do_symmetrize: bool,
    early_stop_rounds: int,
    n_trials: int = 150,
) -> dict:
    """Phase 1: Tune LightGBM params, evaluated through meta-learner pipeline.

    Uses fast settings during search: small ensemble size and fewer RF trees.

    Returns:
        Best LightGBM hyperparameters dict.
    """
    # Fast meta config for search speed
    fast_meta_cfg = {
        **meta_cfg,
        "lgbm_oof_ensemble_size": 3,
        "rf_n_estimators": 50,
    }

    available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())
    valid_folds = [
        f for f in cv_folds if f["val_year"] in available_years
    ]

    logger.info(
        "Phase 1: Tuning LightGBM through meta-learner (%d trials, %d folds)",
        n_trials, len(valid_folds),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.3, log=True,
            ),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 100,
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0,
            ),
            "min_split_gain": trial.suggest_float(
                "min_split_gain", 0.0, 1.0,
            ),
        }

        fold_losses = []
        for step, fold in enumerate(valid_folds):
            fold_loss = _evaluate_meta_pipeline_on_fold(
                matchups_df=matchups_df,
                train_end=fold["train_end"],
                val_year=fold["val_year"],
                lgbm_params=params,
                meta_cfg=fast_meta_cfg,
                feature_mode=feature_mode,
                do_symmetrize=do_symmetrize,
                early_stop_rounds=early_stop_rounds,
            )
            fold_losses.append(fold_loss)
            trial.report(float(np.mean(fold_losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_loss = float(np.mean(fold_losses))
        logger.info(
            "Trial %d: mean meta log_loss=%.4f", trial.number, mean_loss,
        )
        return mean_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best["objective"] = "binary"
    best["metric"] = "binary_logloss"

    logger.info(
        "Phase 1 complete. Best meta log_loss: %.4f", study.best_value,
    )
    logger.info("Best LightGBM params: %s", best)

    return best


# ---------------------------------------------------------------------------
# Phase 2: Tune meta-learner params
# ---------------------------------------------------------------------------


def run_phase2_meta_tuning(
    matchups_df: pd.DataFrame,
    cv_folds: list[dict],
    frozen_lgbm_params: dict,
    base_meta_cfg: dict,
    feature_mode: str,
    do_symmetrize: bool,
    early_stop_rounds: int,
    n_trials: int = 100,
) -> dict:
    """Phase 2: Tune meta-learner params with frozen LightGBM.

    Search space: lr_C, rf_n_estimators, rf_max_depth, rf_min_samples_leaf,
    meta_C, lgbm_oof_ensemble_size.

    Returns:
        Best meta-learner configuration dict.
    """
    available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())
    valid_folds = [
        f for f in cv_folds if f["val_year"] in available_years
    ]

    logger.info(
        "Phase 2: Tuning meta-learner params (%d trials, %d folds)",
        n_trials, len(valid_folds),
    )

    def objective(trial: optuna.Trial) -> float:
        meta_cfg = {
            **base_meta_cfg,
            "lr_C": trial.suggest_float("lr_C", 0.001, 100.0, log=True),
            "rf_n_estimators": trial.suggest_int(
                "rf_n_estimators", 50, 500, step=50,
            ),
            "rf_max_depth": trial.suggest_int("rf_max_depth", 3, 12),
            "rf_min_samples_leaf": trial.suggest_int(
                "rf_min_samples_leaf", 5, 50,
            ),
            "meta_C": trial.suggest_float("meta_C", 0.001, 100.0, log=True),
            "lgbm_oof_ensemble_size": trial.suggest_int(
                "lgbm_oof_ensemble_size", 3, 20,
            ),
        }

        fold_losses = []
        for step, fold in enumerate(valid_folds):
            fold_loss = _evaluate_meta_pipeline_on_fold(
                matchups_df=matchups_df,
                train_end=fold["train_end"],
                val_year=fold["val_year"],
                lgbm_params=frozen_lgbm_params,
                meta_cfg=meta_cfg,
                feature_mode=feature_mode,
                do_symmetrize=do_symmetrize,
                early_stop_rounds=early_stop_rounds,
            )
            fold_losses.append(fold_loss)
            trial.report(float(np.mean(fold_losses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_loss = float(np.mean(fold_losses))
        logger.info(
            "Trial %d: mean meta log_loss=%.4f", trial.number, mean_loss,
        )
        return mean_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    best_meta = {
        **base_meta_cfg,
        **study.best_params,
    }

    logger.info(
        "Phase 2 complete. Best meta log_loss: %.4f", study.best_value,
    )
    logger.info("Best meta-learner config: %s", best_meta)

    return best_meta


# ---------------------------------------------------------------------------
# Save tuned params to config
# ---------------------------------------------------------------------------


def save_e2e_params(
    lgbm_params: dict,
    meta_cfg: dict,
    config_path: Path | None = None,
) -> None:
    """Save Phase 1 + Phase 2 tuned params to model_config.yaml.

    Updates both hyperparameters (LightGBM) and meta_learner sections.
    """
    config_path = config_path or config.MODEL_CONFIG_FILE

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    mm = dict(full_config["march_madness"])

    # Update LightGBM hyperparameters
    hp = dict(mm.get("hyperparameters", {}))
    for k, v in lgbm_params.items():
        if k not in ("objective", "metric"):
            hp[k] = v
    mm["hyperparameters"] = hp

    # Update meta-learner config
    ml = dict(mm.get("meta_learner", {}))
    for k in (
        "lr_C", "rf_n_estimators", "rf_max_depth",
        "rf_min_samples_leaf", "meta_C", "lgbm_oof_ensemble_size",
    ):
        if k in meta_cfg:
            ml[k] = meta_cfg[k]
    mm["meta_learner"] = ml

    new_config = dict(full_config)
    new_config["march_madness"] = mm

    with open(config_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved e2e tuned params to %s", config_path)
