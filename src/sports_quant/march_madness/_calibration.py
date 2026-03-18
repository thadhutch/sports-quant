"""Probability calibration for March Madness predictions.

Provides isotonic regression calibration fitted on out-of-fold predictions,
with probability clipping to prevent infinite log loss.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import early_stopping as es_callback
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sports_quant.march_madness._config import build_lgbm
from sports_quant.march_madness._data import (
    compute_difference_features,
    symmetrize_training_data,
)
from sports_quant.march_madness._features import TARGET_COLUMN, YEAR_COLUMN

logger = logging.getLogger(__name__)


def fit_calibrator(
    oof_probabilities: np.ndarray,
    oof_labels: np.ndarray,
    method: str = "isotonic",
) -> IsotonicRegression | LogisticRegression:
    """Fit a calibration model on out-of-fold predictions.

    Args:
        oof_probabilities: Raw model probabilities from temporal CV folds.
        oof_labels: Actual binary outcomes.
        method: "isotonic" for IsotonicRegression, "platt" for Platt scaling.

    Returns:
        Fitted calibrator.

    Raises:
        ValueError: If inputs are invalid or method is unknown.
    """
    if len(oof_probabilities) != len(oof_labels):
        raise ValueError(
            f"Shape mismatch: probabilities ({len(oof_probabilities)}) "
            f"vs labels ({len(oof_labels)})"
        )

    if len(oof_probabilities) < 10:
        raise ValueError(
            f"Too few samples for calibration: {len(oof_probabilities)}. "
            f"Need at least 10."
        )

    if method == "isotonic":
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip",
        )
        calibrator.fit(oof_probabilities, oof_labels)
    elif method == "platt":
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(
            oof_probabilities.reshape(-1, 1), oof_labels,
        )
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    logger.info(
        "Fitted %s calibrator on %d samples",
        method, len(oof_probabilities),
    )
    return calibrator


def calibrate_probabilities(
    calibrator: IsotonicRegression | LogisticRegression,
    raw_probabilities: np.ndarray,
    clip_min: float = 0.025,
    clip_max: float = 0.975,
) -> np.ndarray:
    """Apply calibration and clip to safe probability range.

    Args:
        calibrator: Fitted calibration model.
        raw_probabilities: Uncalibrated model probabilities.
        clip_min: Minimum probability (prevents infinite log loss).
        clip_max: Maximum probability.

    Returns:
        Calibrated and clipped probabilities.
    """
    if isinstance(calibrator, IsotonicRegression):
        calibrated = calibrator.predict(raw_probabilities)
    else:
        # Platt scaling: use predict_proba
        calibrated = calibrator.predict_proba(
            raw_probabilities.reshape(-1, 1),
        )[:, 1]

    return np.clip(calibrated, clip_min, clip_max)


def collect_oof_predictions(
    matchups_df: pd.DataFrame,
    cv_folds: list[dict],
    hyperparams: dict,
    do_symmetrize: bool = True,
    early_stop_rounds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect out-of-fold predictions from temporal CV for calibration.

    Trains a model on each fold's training data and predicts on the
    validation fold. Concatenates all fold predictions.

    Args:
        matchups_df: Full matchup DataFrame with raw columns.
        cv_folds: List of {train_end, val_year} dicts.
        hyperparams: LightGBM hyperparameters from config.
        do_symmetrize: Whether to symmetrize training data.
        early_stop_rounds: Early stopping patience.

    Returns:
        Tuple of (oof_probabilities, oof_labels) as numpy arrays.
    """
    available_years = set(matchups_df[YEAR_COLUMN].unique())
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for fold in cv_folds:
        val_year = fold["val_year"]
        train_end = fold["train_end"]

        if train_end >= val_year:
            raise ValueError(
                f"Data leakage: train_end ({train_end}) >= val_year "
                f"({val_year}). train_end must be strictly less than "
                f"val_year to prevent validation data from leaking "
                f"into training."
            )

        if val_year not in available_years:
            logger.warning("Skipping fold val_year=%d (no data)", val_year)
            continue

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

        model = build_lgbm(hyperparams, random_seed=42)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["binary_logloss"],
            callbacks=[es_callback(early_stop_rounds)],
        )

        y_val_proba = model.predict_proba(X_val)[:, 1]
        all_probs.append(y_val_proba)
        all_labels.append(y_val.to_numpy())

        logger.info(
            "OOF fold val_year=%d: %d samples, mean prob=%.3f",
            val_year, len(y_val), y_val_proba.mean(),
        )

    if not all_probs:
        raise ValueError("No OOF predictions collected from any fold")

    return np.concatenate(all_probs), np.concatenate(all_labels)
