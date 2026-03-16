"""Optuna hyperparameter tuning for March Madness LightGBM models.

Runs Bayesian optimization with temporal cross-validation folds,
using log loss as the objective.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna
import yaml
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import log_loss

from sports_quant import _config as config
from sports_quant.march_madness._config import load_mm_config
from sports_quant.march_madness._data import (
    compute_difference_features,
    symmetrize_training_data,
)
from sports_quant.march_madness._features import TARGET_COLUMN, YEAR_COLUMN

logger = logging.getLogger(__name__)


def run_optuna_study(
    matchups_df: None | "pd.DataFrame" = None,
    n_trials: int | None = None,
) -> dict:
    """Run Bayesian hyperparameter optimization with temporal CV.

    Args:
        matchups_df: Full matchup DataFrame. If None, loads from CSV.
        n_trials: Number of Optuna trials. If None, uses config value.

    Returns:
        Dict of best hyperparameters.
    """
    import pandas as pd
    from sports_quant.march_madness import _config as mm_config

    cfg = load_mm_config()
    cv_folds = cfg.get("tuning", {}).get("cv_folds", [])
    n_trials = n_trials or cfg.get("tuning", {}).get("n_trials", 100)
    do_symmetrize = cfg.get("train", {}).get("symmetrize", False)
    early_stop_rounds = cfg.get("train", {}).get("early_stopping_rounds", 50)

    if matchups_df is None:
        matchups_df = pd.read_csv(mm_config.MM_TRAINING_DATA)

    # Validate folds exist in data
    available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())
    valid_folds = [
        fold for fold in cv_folds
        if fold["val_year"] in available_years
    ]

    if not valid_folds:
        raise ValueError(
            f"No valid CV folds. Available years: {available_years}, "
            f"Configured folds: {cv_folds}"
        )

    logger.info(
        "Starting Optuna study: %d trials, %d CV folds",
        n_trials, len(valid_folds),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 8, 32),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True,
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 100,
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

        fold_losses: list[float] = []

        for fold in valid_folds:
            train_end = fold["train_end"]
            val_year = fold["val_year"]

            train_raw = matchups_df[matchups_df[YEAR_COLUMN] <= train_end]
            val_raw = matchups_df[matchups_df[YEAR_COLUMN] == val_year]

            if len(val_raw) == 0:
                continue

            X_train = compute_difference_features(train_raw)
            y_train = train_raw[TARGET_COLUMN].reset_index(drop=True)
            X_val = compute_difference_features(val_raw)
            y_val = val_raw[TARGET_COLUMN].reset_index(drop=True)

            if do_symmetrize:
                X_train, y_train = symmetrize_training_data(X_train, y_train)

            model = LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                random_state=42,
                verbose=-1,
                **params,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=["binary_logloss"],
                callbacks=[early_stopping(early_stop_rounds)],
            )

            y_val_proba = model.predict_proba(X_val)[:, 1]
            fold_loss = log_loss(y_val, y_val_proba)
            fold_losses.append(fold_loss)

        if not fold_losses:
            return float("inf")

        return float(np.mean(fold_losses))

    # Suppress Optuna's default logging (use our logger instead)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        "Optuna study complete. Best log loss: %.4f",
        study.best_value,
    )
    logger.info("Best params: %s", study.best_params)

    return study.best_params


def save_best_params(params: dict, config_path: Path | None = None) -> None:
    """Save tuned hyperparameters to model_config.yaml.

    Merges the new params into the existing hyperparameters section
    without overwriting objective/metric.

    Args:
        params: Dict of hyperparameter names to values from Optuna.
        config_path: Path to config file. Defaults to MODEL_CONFIG_FILE.
    """
    config_path = config_path or config.MODEL_CONFIG_FILE

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    # Create new config with updated params (immutable pattern)
    mm_cfg = dict(full_config["march_madness"])
    hp = dict(mm_cfg.get("hyperparameters", {}))
    hp.update(params)
    mm_cfg["hyperparameters"] = hp

    new_config = dict(full_config)
    new_config["march_madness"] = mm_cfg

    with open(config_path, "w") as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved best params to %s", config_path)
