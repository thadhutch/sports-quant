"""Single-model and ensemble training logic for O/U prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sports_quant.modeling._data import DATE_COLUMN, TARGET_COLUMN
from sports_quant.modeling._features import ALL_FEATURES

logger = logging.getLogger(__name__)

# Confidence bins: 50% to 100% in 5-point increments
CONF_BINS = np.arange(0.5, 1.05, 0.05)
CONF_LABELS = [f"{int(b * 100)}-{int((b + 0.05) * 100)}%" for b in CONF_BINS[:-1]]


@dataclass
class TrainedModel:
    """Container for a single trained XGBoost model and its validation metrics."""

    model: XGBClassifier
    overall_accuracy: float
    current_season_accuracy: float
    last_season_accuracy: float
    confidence_accuracy: pd.DataFrame = field(repr=False)


def _build_classifier(
    seed: int,
    hyperparameters: dict | None = None,
) -> XGBClassifier:
    """Create an XGBClassifier with the given seed and optional hyperparameters."""
    defaults = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }
    if hyperparameters:
        defaults.update(hyperparameters)

    return XGBClassifier(
        random_state=seed,
        verbosity=0,
        **defaults,
    )


def train_ensemble_for_date(
    df: pd.DataFrame,
    current_date,
    date_index: int,
    *,
    n_models: int = 50,
    test_size: float = 0.2,
    accuracy_threshold: float = 0.50,
    hyperparameters: dict | None = None,
) -> list[TrainedModel]:
    """Train *n_models* XGBoost models for a single game-day.

    Only models with overall validation accuracy above *accuracy_threshold*
    are returned.

    This mirrors ``nfl-model/algorithm.py`` lines 208-376.
    """
    train_df = df[df[DATE_COLUMN] < current_date]
    test_df = df[df[DATE_COLUMN] == current_date]

    if test_df.empty:
        return []

    train_seasons = train_df["season"].unique()
    if len(train_seasons) < 2:
        logger.warning(
            "Skipping %s — fewer than 2 training seasons.", current_date
        )
        return []

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    X_full = train_df[ALL_FEATURES].reset_index(drop=True)
    y_full = train_df[TARGET_COLUMN].reset_index(drop=True)
    seasons_full = train_df["season"].reset_index(drop=True)

    models: list[TrainedModel] = []

    for model_idx in range(n_models):
        seed = 42 + model_idx + date_index * 1000

        try:
            X_train, X_val, y_train, y_val, _, seasons_val = train_test_split(
                X_full,
                y_full,
                seasons_full,
                test_size=test_size,
                random_state=seed,
                stratify=y_full if len(np.unique(y_full)) > 1 else None,
            )
        except ValueError as exc:
            logger.error("train_test_split failed (model %d): %s", model_idx + 1, exc)
            continue

        clf = _build_classifier(seed, hyperparameters)
        try:
            clf.fit(X_train, y_train)
        except Exception as exc:
            logger.error("Training failed (model %d): %s", model_idx + 1, exc)
            continue

        # Validation metrics
        y_val_pred = clf.predict(X_val)
        y_val_proba = clf.predict_proba(X_val)
        val_conf = np.max(y_val_proba, axis=1)

        overall_acc = accuracy_score(y_val, y_val_pred)
        if overall_acc <= accuracy_threshold:
            continue

        # Per-season accuracy
        current_season = seasons_val.max()
        last_season = current_season - 1

        current_mask = seasons_val == current_season
        current_acc = (
            accuracy_score(y_val[current_mask], y_val_pred[current_mask])
            if current_mask.any()
            else 0.0
        )

        last_mask = seasons_val == last_season
        last_acc = (
            accuracy_score(y_val[last_mask], y_val_pred[last_mask])
            if last_mask.any()
            else 0.0
        )

        # Accuracy by confidence bin and season
        val_results = pd.DataFrame(
            {
                "Actual": y_val.values,
                "Predicted": y_val_pred,
                "Confidence": val_conf,
                "Season": seasons_val.values,
            }
        )
        val_results["Confidence Bin"] = pd.cut(
            val_results["Confidence"],
            bins=CONF_BINS,
            labels=CONF_LABELS,
            include_lowest=True,
        )
        val_results["Correct"] = (val_results["Actual"] == val_results["Predicted"]).astype(int)

        conf_acc = (
            val_results.groupby(["Confidence Bin", "Season"], observed=False)["Correct"]
            .mean()
            .reset_index(name="Accuracy")
        )

        models.append(
            TrainedModel(
                model=clf,
                overall_accuracy=overall_acc,
                current_season_accuracy=current_acc,
                last_season_accuracy=last_acc,
                confidence_accuracy=conf_acc,
            )
        )
        logger.debug(
            "Model %d: overall=%.4f current=%.4f last=%.4f",
            model_idx + 1,
            overall_acc,
            current_acc,
            last_acc,
        )

    logger.info(
        "%s: %d / %d models passed threshold (%.0f%%)",
        current_date,
        len(models),
        n_models,
        accuracy_threshold * 100,
    )
    return models


def train_backtest_model_for_date(
    df: pd.DataFrame,
    current_date,
    model_idx: int,
    *,
    test_size: float = 0.8,
    hyperparameters: dict | None = None,
) -> tuple[list, list, list, list, list] | None:
    """Train one walk-forward backtest model for a single date.

    Uses only data before *current_date* for training and the current date
    for testing.  A random *test_size* fraction of the training data is
    discarded to introduce model diversity (matching ``backtest.py``).

    Returns ``(predictions, actuals, dates, seasons, confidences)`` or
    ``None`` on failure.
    """
    seed = 42 + model_idx

    train_df = df[df[DATE_COLUMN] < current_date]
    test_df = df[df[DATE_COLUMN] == current_date]

    if train_df.empty or test_df.empty:
        return None

    train_seasons = train_df["season"].unique()
    if len(train_seasons) < 2:
        return None

    X_train_full = train_df[ALL_FEATURES]
    y_train_full = train_df[TARGET_COLUMN]

    try:
        X_train, _, y_train, _ = train_test_split(
            X_train_full,
            y_train_full,
            test_size=test_size,
            random_state=seed,
            stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None,
        )
    except ValueError as exc:
        logger.error("train_test_split failed (backtest model %d): %s", model_idx + 1, exc)
        return None

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    if X_train.empty or X_test.empty:
        return None

    clf = _build_classifier(seed, hyperparameters)
    try:
        clf.fit(X_train, y_train)
    except Exception as exc:
        logger.error("Training failed (backtest model %d): %s", model_idx + 1, exc)
        return None

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    confs = np.max(y_proba, axis=1).tolist()

    return (
        y_pred.tolist(),
        y_test.tolist(),
        test_df[DATE_COLUMN].tolist(),
        test_df["season"].tolist(),
        confs,
    )
