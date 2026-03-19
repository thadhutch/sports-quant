"""Stacking meta-learner for March Madness predictions.

Trains diverse base learners (LightGBM ensemble, logistic regression,
random forest), collects out-of-fold predictions via temporal CV,
and trains a meta-learner to optimally combine them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import early_stopping as es_callback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sports_quant.march_madness._config import build_lgbm
from sports_quant.march_madness._data import symmetrize_training_data
from sports_quant.march_madness._features import TARGET_COLUMN, YEAR_COLUMN

logger = logging.getLogger(__name__)

BASE_LEARNER_NAMES = ("lgbm_ensemble", "logistic_regression", "random_forest")


@dataclass(frozen=True)
class StackedOOF:
    """Out-of-fold predictions from all base learners."""

    matrix: np.ndarray  # (n_samples, n_base_learners)
    labels: np.ndarray  # (n_samples,)
    names: tuple[str, ...]


@dataclass(frozen=True)
class TrainedStack:
    """A trained meta-learner with its base learner predictions."""

    meta_model: LogisticRegression
    base_predictions: np.ndarray  # (n_test, n_base_learners)
    meta_predictions: np.ndarray  # (n_test,)
    names: tuple[str, ...]


# ---------------------------------------------------------------------------
# Base learner factories
# ---------------------------------------------------------------------------


def _build_lr(meta_cfg: dict) -> Pipeline:
    """Build an imputed + scaled logistic regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=meta_cfg.get("lr_C", 1.0),
            solver="lbfgs",
            max_iter=1000,
        )),
    ])


def _build_rf(meta_cfg: dict) -> Pipeline:
    """Build an imputed random forest pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=meta_cfg.get("rf_n_estimators", 200),
            max_depth=meta_cfg.get("rf_max_depth", 6),
            min_samples_leaf=meta_cfg.get("rf_min_samples_leaf", 20),
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Base learner training helpers
# ---------------------------------------------------------------------------


def _train_lgbm_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    hyperparams: dict,
    n_models: int,
    early_stop_rounds: int,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> np.ndarray:
    """Train N LightGBM models and return averaged predictions."""
    all_probas = []
    rng = np.random.RandomState(42)

    for _ in range(n_models):
        seed = int(rng.randint(1, 10000))
        model = build_lgbm(hyperparams, seed)

        if X_val is not None and len(X_val) > 0:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=["binary_logloss"],
                callbacks=[es_callback(early_stop_rounds)],
            )
        else:
            model.fit(X_train, y_train)

        all_probas.append(model.predict_proba(X_pred)[:, 1])

    return np.mean(all_probas, axis=0)


def _train_sklearn_predict(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
) -> np.ndarray:
    """Train a sklearn-compatible model and return probability predictions."""
    model.fit(X_train, y_train)
    return model.predict_proba(X_pred)[:, 1]


# ---------------------------------------------------------------------------
# OOF collection
# ---------------------------------------------------------------------------


def collect_stacked_oof(
    matchups_df: pd.DataFrame,
    prior_years: list[int],
    prepare_features_fn,
    hyperparams: dict,
    meta_cfg: dict,
    do_symmetrize: bool,
    feature_mode: str,
    early_stop_rounds: int,
) -> StackedOOF:
    """Collect OOF predictions from all base learners via temporal CV.

    For each year in prior_years (with enough training history), trains
    all base learners on earlier years and predicts on that year.

    Args:
        matchups_df: Full matchups DataFrame (all years).
        prior_years: Sorted list of years available for OOF.
        prepare_features_fn: Function(df, feature_mode) -> feature DataFrame.
        hyperparams: LightGBM hyperparameters.
        meta_cfg: Meta-learner configuration dict.
        do_symmetrize: Whether to symmetrize training data.
        feature_mode: Feature extraction mode.
        early_stop_rounds: Early stopping patience.

    Returns:
        StackedOOF with concatenated OOF predictions from all folds.
    """
    min_train_years = meta_cfg.get("min_oof_years", 3)
    n_lgbm = meta_cfg.get("lgbm_oof_ensemble_size", 10)

    oof_by_learner: dict[str, list[np.ndarray]] = {
        name: [] for name in BASE_LEARNER_NAMES
    }
    all_labels: list[np.ndarray] = []

    for val_year in prior_years:
        train_years = [y for y in prior_years if y < val_year]
        if len(train_years) < min_train_years:
            continue

        train_df = matchups_df[matchups_df[YEAR_COLUMN].isin(train_years)]
        val_df = matchups_df[matchups_df[YEAR_COLUMN] == val_year]

        if len(val_df) == 0:
            continue

        X_train = prepare_features_fn(train_df, feature_mode)
        y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
        X_val = prepare_features_fn(val_df, feature_mode)
        y_val = val_df[TARGET_COLUMN].reset_index(drop=True)

        if do_symmetrize and feature_mode in ("difference", "combined"):
            X_train_s, y_train_s = symmetrize_training_data(X_train, y_train)
        else:
            X_train_s, y_train_s = X_train, y_train

        # LightGBM ensemble (smaller for OOF speed)
        lgbm_preds = _train_lgbm_ensemble(
            X_train_s, y_train_s, X_val, hyperparams,
            n_models=n_lgbm,
            early_stop_rounds=early_stop_rounds,
        )
        oof_by_learner["lgbm_ensemble"].append(lgbm_preds)

        # Logistic Regression
        lr_preds = _train_sklearn_predict(
            _build_lr(meta_cfg), X_train_s, y_train_s, X_val,
        )
        oof_by_learner["logistic_regression"].append(lr_preds)

        # Random Forest
        rf_preds = _train_sklearn_predict(
            _build_rf(meta_cfg), X_train_s, y_train_s, X_val,
        )
        oof_by_learner["random_forest"].append(rf_preds)

        all_labels.append(y_val.to_numpy())

        logger.info(
            "OOF fold val_year=%d: %d samples (trained on %d years)",
            val_year, len(y_val), len(train_years),
        )

    if not all_labels:
        raise ValueError("No OOF predictions collected — not enough years")

    matrix = np.column_stack([
        np.concatenate(oof_by_learner[name]) for name in BASE_LEARNER_NAMES
    ])

    logger.info(
        "Stacked OOF: %d samples x %d base learners",
        matrix.shape[0], matrix.shape[1],
    )

    return StackedOOF(
        matrix=matrix,
        labels=np.concatenate(all_labels),
        names=BASE_LEARNER_NAMES,
    )


# ---------------------------------------------------------------------------
# Meta-learner training
# ---------------------------------------------------------------------------


def train_meta_learner(
    oof: StackedOOF,
    meta_cfg: dict,
) -> LogisticRegression:
    """Train the meta-learner on stacked OOF predictions.

    Uses logistic regression — simple enough to avoid overfitting on the
    small stacked feature space (3 features).
    """
    meta_C = meta_cfg.get("meta_C", 1.0)
    meta = LogisticRegression(C=meta_C, solver="lbfgs", max_iter=1000)
    meta.fit(oof.matrix, oof.labels)

    coef_dict = {
        name: round(float(c), 4)
        for name, c in zip(oof.names, meta.coef_[0])
    }
    logger.info(
        "Meta-learner trained on %d samples, %d base learners. "
        "Coefficients: %s, Intercept: %.4f",
        len(oof.labels), oof.matrix.shape[1],
        coef_dict, float(meta.intercept_[0]),
    )

    return meta


# ---------------------------------------------------------------------------
# End-to-end stacking for backtest
# ---------------------------------------------------------------------------


def train_and_predict_stack(
    matchups_df: pd.DataFrame,
    backtest_year: int,
    prepare_features_fn,
    hyperparams: dict,
    meta_cfg: dict,
    do_symmetrize: bool,
    feature_mode: str,
    early_stop_rounds: int,
    lgbm_backtest_probas: np.ndarray,
) -> TrainedStack:
    """End-to-end: collect OOF, train meta-learner, predict on backtest.

    Reuses the pre-computed LightGBM ensemble predictions from the main
    training loop as one base learner input. Trains logistic regression
    and random forest on ALL prior years for the other base inputs.

    Args:
        matchups_df: Full matchups DataFrame (all years).
        backtest_year: Year being backtested.
        prepare_features_fn: Function(df, feature_mode) -> feature DataFrame.
        hyperparams: LightGBM hyperparameters.
        meta_cfg: Meta-learner configuration dict.
        do_symmetrize: Whether to symmetrize training data.
        feature_mode: Feature extraction mode.
        early_stop_rounds: Early stopping patience.
        lgbm_backtest_probas: Pre-computed LightGBM ensemble predictions
            on backtest data (averaged from the main training loop).

    Returns:
        TrainedStack with meta-learner predictions on the backtest year.
    """
    available_years = sorted(matchups_df[YEAR_COLUMN].unique().tolist())
    prior_years = [y for y in available_years if y < backtest_year]

    # Step 1: Collect OOF predictions via temporal CV
    oof = collect_stacked_oof(
        matchups_df=matchups_df,
        prior_years=prior_years,
        prepare_features_fn=prepare_features_fn,
        hyperparams=hyperparams,
        meta_cfg=meta_cfg,
        do_symmetrize=do_symmetrize,
        feature_mode=feature_mode,
        early_stop_rounds=early_stop_rounds,
    )

    # Step 2: Train meta-learner
    meta_model = train_meta_learner(oof, meta_cfg)

    # Step 3: Train LR and RF on ALL prior data, predict on backtest
    all_prior_df = matchups_df[matchups_df[YEAR_COLUMN].isin(prior_years)]
    backtest_df = matchups_df[matchups_df[YEAR_COLUMN] == backtest_year]

    X_all_prior = prepare_features_fn(all_prior_df, feature_mode)
    y_all_prior = all_prior_df[TARGET_COLUMN].reset_index(drop=True)
    X_backtest = prepare_features_fn(backtest_df, feature_mode)

    if do_symmetrize and feature_mode in ("difference", "combined"):
        X_all_prior, y_all_prior = symmetrize_training_data(
            X_all_prior, y_all_prior,
        )

    lr_preds = _train_sklearn_predict(
        _build_lr(meta_cfg), X_all_prior, y_all_prior, X_backtest,
    )
    rf_preds = _train_sklearn_predict(
        _build_rf(meta_cfg), X_all_prior, y_all_prior, X_backtest,
    )

    # Step 4: Stack base predictions and run meta-learner
    base_preds = np.column_stack([lgbm_backtest_probas, lr_preds, rf_preds])
    meta_preds = meta_model.predict_proba(base_preds)[:, 1]

    logger.info(
        "Meta-learner predictions for %d: mean=%.3f, "
        "base means=[lgbm=%.3f, lr=%.3f, rf=%.3f]",
        backtest_year, meta_preds.mean(),
        lgbm_backtest_probas.mean(), lr_preds.mean(), rf_preds.mean(),
    )

    return TrainedStack(
        meta_model=meta_model,
        base_predictions=base_preds,
        meta_predictions=meta_preds,
        names=BASE_LEARNER_NAMES,
    )
