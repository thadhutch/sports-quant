"""Train 50 LightGBM models for 2026 prediction.

Follows the same pipeline as run_backtest() but only trains and saves
models — no evaluation step (since 2026 results don't exist yet).

Training data: all years through 2025.
Validation: 2024 + 2025 (2 most recent years, matching backtest config).
"""

import logging

import joblib
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.metrics import f1_score

from sports_quant.march_madness._config import (
    MM_BACKTEST_DIR,
    build_lgbm,
    load_mm_config,
)
from sports_quant.march_madness._data import (
    compute_combined_difference_features,
    compute_difference_features,
    load_and_prepare,
    symmetrize_training_data,
)
from sports_quant.march_madness._features import (
    DROP_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _prepare_features(df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    if feature_mode == "combined":
        return compute_combined_difference_features(df)
    if feature_mode == "difference":
        return compute_difference_features(df)
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    filtered = df.drop(columns=drop_cols)
    return filtered.drop(columns=[TARGET_COLUMN], errors="ignore")


def main() -> None:
    target_year = 2026
    cfg = load_mm_config()
    model_version = cfg["model_version"]
    num_models = cfg["models_to_train"]
    hyperparams = cfg["hyperparameters"]
    feature_mode = cfg.get("feature_mode", "raw")
    do_symmetrize = cfg.get("train", {}).get("symmetrize", False)
    early_stop_rounds = cfg.get("train", {}).get("early_stopping_rounds", 50)
    val_n_years = cfg.get("backtest", {}).get("val_years", 2)
    min_boosting_rounds = cfg.get("train", {}).get("min_boosting_rounds", 50)

    models_dir = MM_BACKTEST_DIR / model_version / str(target_year) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset
    data = load_and_prepare(feature_mode="raw")
    matchups = data.df

    all_years = sorted(matchups[YEAR_COLUMN].unique().tolist())
    all_prior = matchups[matchups[YEAR_COLUMN] < target_year]

    # Temporal validation: most recent N years before 2026
    candidates = sorted(
        [y for y in all_years if y < target_year], reverse=True,
    )
    val_years = candidates[:val_n_years]

    if val_years:
        earliest_val = min(val_years)
        train_data = all_prior[all_prior[YEAR_COLUMN] < earliest_val]
        val_data = all_prior[all_prior[YEAR_COLUMN].isin(val_years)]
    else:
        train_data = all_prior
        val_data = pd.DataFrame(columns=all_prior.columns)

    logger.info(
        "Training: %d games (years < %d), Val: %d games (years %s)",
        len(train_data),
        earliest_val if val_years else target_year,
        len(val_data), val_years,
    )

    # Prepare features
    X_train = _prepare_features(train_data, feature_mode)
    y_train = train_data[TARGET_COLUMN].reset_index(drop=True)

    X_val = _prepare_features(val_data, feature_mode) if len(val_data) > 0 else None
    y_val = val_data[TARGET_COLUMN].reset_index(drop=True) if len(val_data) > 0 else None

    # Symmetrize training data
    if do_symmetrize and feature_mode in ("difference", "combined"):
        X_train, y_train = symmetrize_training_data(X_train, y_train)
        logger.info("Symmetrized training data: %d rows", len(X_train))

    # Train N models
    has_val = X_val is not None and len(X_val) > 0
    all_models = []

    for model_num in range(num_models):
        random_seed = np.random.randint(1, 10000)
        model = build_lgbm(hyperparams, random_seed)

        if has_val:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            callbacks = [
                early_stopping(early_stop_rounds, first_metric_only=True),
            ]
        else:
            eval_set = [(X_train, y_train)]
            callbacks = None

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=["binary_logloss", "auc"],
            callbacks=callbacks,
        )

        # Guard against under-trained models
        actual_iters = model.best_iteration_ if has_val else model.n_estimators
        if has_val and actual_iters < min_boosting_rounds:
            logger.warning(
                "Model %d stopped at %d iterations (min=%d). Retraining.",
                model_num, actual_iters, min_boosting_rounds,
            )
            model = build_lgbm(
                {**hyperparams, "n_estimators": min_boosting_rounds},
                random_seed,
            )
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric=["binary_logloss", "auc"],
            )
            actual_iters = min_boosting_rounds

        # Compute val F1 for ranking
        val_f1 = 0.0
        if has_val:
            y_val_pred = model.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred)

        all_models.append({
            "model": model,
            "val_f1": val_f1,
            "seed": random_seed,
            "iterations": actual_iters,
        })

        logger.info(
            "Model %d/%d: val_f1=%.4f, iters=%d",
            model_num + 1, num_models, val_f1, actual_iters,
        )

    # Rank by val F1 and save all models
    ranked = sorted(all_models, key=lambda x: x["val_f1"], reverse=True)

    for i, md in enumerate(ranked):
        path = models_dir / f"top_model_{i + 1}.joblib"
        joblib.dump(md["model"], path)

    mean_f1 = np.mean([m["val_f1"] for m in ranked])
    logger.info(
        "Saved %d models to %s (mean val F1: %.4f)",
        len(ranked), models_dir, mean_f1,
    )

    # Print top 5
    print(f"\nTop 5 models by validation F1:")
    for i, md in enumerate(ranked[:5]):
        print(f"  #{i+1}: val_f1={md['val_f1']:.4f}, iters={md['iterations']}")


if __name__ == "__main__":
    main()
