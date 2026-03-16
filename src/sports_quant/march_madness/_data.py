"""Data loading and preparation for March Madness models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._features import (
    DIFF_FEATURE_COLUMNS,
    DROP_COLUMNS,
    STAT_PAIRS,
    TARGET_COLUMN,
    TEAM_INFO_COLUMNS,
    YEAR_COLUMN,
)

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Container for model-ready March Madness data."""

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    years: list[int]
    team_info: pd.DataFrame


def compute_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise difference features from raw Team1/Team2 columns.

    Transforms 36 raw KenPom columns into 21 difference features:
    18 stat diffs + seed_diff + 2 derived features.

    Args:
        df: DataFrame with raw Team1 and Team2 columns plus Seed1/Seed2.

    Returns:
        New DataFrame with only DIFF_FEATURE_COLUMNS (21 columns).
    """
    result: dict[str, pd.Series] = {}

    # 18 stat differences
    for team1_col, team2_col, diff_name in STAT_PAIRS:
        result[diff_name] = df[team1_col] - df[team2_col]

    # Seed difference
    result["seed_diff"] = df["Seed1"] - df["Seed2"]

    # Derived: efficiency ratio diff = (AdjO/AdjD)_team1 - (AdjO/AdjD)_team2
    ratio_team1 = df["AdjustO"] / df["AdjustD"]
    ratio_team2 = df["AdjustO_Team2"] / df["AdjustD_Team2"]
    result["efficiency_ratio_diff"] = ratio_team1 - ratio_team2

    # Derived: seed x adjEM interaction
    result["seed_x_adjEM_interaction"] = (
        result["seed_diff"] * result["adjEM_diff"]
    )

    return pd.DataFrame(result, columns=list(DIFF_FEATURE_COLUMNS))


def symmetrize_training_data(
    X: pd.DataFrame, y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Double training data by swapping team order and flipping labels.

    For difference features, swapping teams negates first-order diffs:
    diff(A,B) = -diff(B,A). Second-order features (products of diffs)
    are even under swap and must be recomputed, not negated.

    Only works with difference features (all columns should be diffs).

    Args:
        X: Feature matrix with difference columns.
        y: Binary target (1 = Team1 wins).

    Returns:
        Tuple of (augmented X, augmented y) with 2x the original rows.

    Raises:
        ValueError: If X columns don't match DIFF_FEATURE_COLUMNS.
    """
    expected = set(DIFF_FEATURE_COLUMNS)
    actual = set(X.columns)
    if actual != expected:
        raise ValueError(
            f"Symmetrization requires difference features. "
            f"Missing: {expected - actual}, Extra: {actual - expected}"
        )

    # Build flipped data without mutating: negate first-order diffs,
    # recompute product feature from the negated components.
    flipped_data = {
        col: (-X[col] if col != "seed_x_adjEM_interaction" else X[col])
        for col in X.columns
    }
    # (-seed_diff) * (-adjEM_diff) = seed_diff * adjEM_diff (same as original)
    flipped_data["seed_x_adjEM_interaction"] = (
        -X["seed_diff"] * -X["adjEM_diff"]
    )
    X_flipped = pd.DataFrame(flipped_data, columns=list(DIFF_FEATURE_COLUMNS))

    y_flipped = 1 - y

    X_sym = pd.concat([X, X_flipped], ignore_index=True)
    y_sym = pd.concat([y, y_flipped], ignore_index=True)

    return X_sym, y_sym


def load_and_prepare(
    feature_mode: str = "difference",
) -> PreparedData:
    """Load training_data.csv and prepare the feature matrix.

    Args:
        feature_mode: "difference" for 21 pairwise diff features,
                      "raw" for original 36 team columns.

    Returns a PreparedData instance with features, target, and metadata.
    """
    df = pd.read_csv(mm_config.MM_TRAINING_DATA)
    logger.info("Loaded %d rows from %s", len(df), mm_config.MM_TRAINING_DATA)

    # Preserve team info before dropping
    info_cols = [c for c in TEAM_INFO_COLUMNS if c in df.columns]
    team_info = df[info_cols].copy()

    years = sorted(df[YEAR_COLUMN].unique().tolist())

    if feature_mode == "difference":
        X = compute_difference_features(df)
    else:
        # Original raw feature mode
        df_filtered = df.drop(
            columns=[c for c in DROP_COLUMNS if c in df.columns]
        )
        X = df_filtered.drop(columns=[TARGET_COLUMN], errors="ignore")

    y = df[TARGET_COLUMN]

    logger.info(
        "Features (%s mode): %d columns, years: %s",
        feature_mode, X.shape[1], years,
    )
    return PreparedData(df=df, X=X, y=y, years=years, team_info=team_info)


def load_prediction_data(
    feature_mode: str = "difference",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prediction_data.csv for inference.

    Args:
        feature_mode: "difference" for 21 pairwise diff features,
                      "raw" for original 36 team columns.

    Returns (X_pred, team_info) where X_pred has features only.
    """
    df = pd.read_csv(mm_config.MM_PREDICTION_DATA)
    logger.info("Loaded %d rows from %s", len(df), mm_config.MM_PREDICTION_DATA)

    # Add Team1_Win as NaN if missing (prediction data won't have outcomes)
    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = np.nan

    # Preserve team info
    team_info = df[
        ["YEAR", "Team1", "Team2", "Seed1", "Seed2", "CURRENT ROUND"]
    ].copy()

    if feature_mode == "difference":
        X_pred = compute_difference_features(df)
    else:
        drop = DROP_COLUMNS + [TARGET_COLUMN]
        X_pred = df.drop(columns=[c for c in drop if c in df.columns])

    return X_pred, team_info
