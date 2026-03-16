"""Data loading and preparation for March Madness models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._features import (
    BARTTORVIK_DERIVED_FEATURES,
    BARTTORVIK_DIFF_FEATURE_COLUMNS,
    BARTTORVIK_STAT_PAIRS,
    COMBINED_DIFF_FEATURE_COLUMNS,
    DIFF_FEATURE_COLUMNS,
    DROP_COLUMNS,
    MATCHUP_EVEN_FEATURES,
    MATCHUP_FEATURE_COLUMNS,
    STAT_PAIRS,
    TARGET_COLUMN,
    TEAM_INFO_COLUMNS,
    YEAR_COLUMN,
    seed_upset_prior,
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


def _has_barttorvik_columns(df: pd.DataFrame) -> bool:
    """Check if the DataFrame contains Barttorvik columns."""
    return "Bart_Rank" in df.columns and "Bart_Rank_Team2" in df.columns


def compute_barttorvik_difference_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise difference features from Barttorvik columns.

    Transforms Barttorvik Team1/Team2 columns into 13 difference features:
    12 stat diffs + 1 derived feature.

    Args:
        df: DataFrame with Barttorvik columns for both teams.

    Returns:
        New DataFrame with only BARTTORVIK_DIFF_FEATURE_COLUMNS (13 columns).
    """
    result: dict[str, pd.Series] = {}

    # 12 stat differences
    for team1_col, team2_col, diff_name in BARTTORVIK_STAT_PAIRS:
        result[diff_name] = df[team1_col] - df[team2_col]

    # Derived: Barttorvik efficiency ratio diff = (AdjOE/AdjDE)_t1 - (AdjOE/AdjDE)_t2
    ratio_t1 = df["Bart_AdjOE"] / df["Bart_AdjDE"]
    ratio_t2 = df["Bart_AdjOE_Team2"] / df["Bart_AdjDE_Team2"]
    result["bart_efficiency_ratio_diff"] = ratio_t1 - ratio_t2

    return pd.DataFrame(result, columns=list(BARTTORVIK_DIFF_FEATURE_COLUMNS))


def compute_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute matchup-specific interaction features.

    Produces 11 features that capture how two teams' styles interact,
    beyond simple stat differences. Requires both KenPom and Barttorvik
    columns plus Seed1/Seed2.

    Args:
        df: DataFrame with KenPom, Barttorvik, and seed columns.

    Returns:
        New DataFrame with MATCHUP_FEATURE_COLUMNS (11 columns).
    """
    result: dict[str, pd.Series] = {}

    # Precompute diffs used by multiple features
    adjO_diff = df["AdjustO"] - df["AdjustO_Team2"]
    adjD_diff = df["AdjustD"] - df["AdjustD_Team2"]
    adjEM_diff = df["AdjEM"] - df["AdjEM_Team2"]
    adjT_diff = df["AdjustT"] - df["AdjustT_Team2"]
    seed_diff = df["Seed1"] - df["Seed2"]
    bart_adjOE_diff = df["Bart_AdjOE"] - df["Bart_AdjOE_Team2"]
    bart_adjDE_diff = df["Bart_AdjDE"] - df["Bart_AdjDE_Team2"]
    sos_adjEM_diff = df["SOS AdjEM"] - df["SOS AdjEM_Team2"]
    bart_barthag_diff = df["Bart_Barthag"] - df["Bart_Barthag_Team2"]

    # Group 1: Offensive vs defensive style interactions
    result["offense_vs_defense_mismatch"] = (
        (df["AdjustO"] - df["AdjustD_Team2"])
        - (df["AdjustO_Team2"] - df["AdjustD"])
    )
    result["bart_offense_vs_defense_mismatch"] = (
        (df["Bart_AdjOE"] - df["Bart_AdjDE_Team2"])
        - (df["Bart_AdjOE_Team2"] - df["Bart_AdjDE"])
    )
    result["offense_defense_product"] = adjO_diff * adjD_diff
    result["bart_offense_defense_product"] = bart_adjOE_diff * bart_adjDE_diff

    # Group 2: Tempo mismatch
    result["tempo_mismatch_magnitude"] = (
        (df["AdjustT"] - df["AdjustT_Team2"]).abs()
    )
    result["tempo_x_quality_interaction"] = adjT_diff * adjEM_diff
    result["tempo_x_seed_interaction"] = adjT_diff * seed_diff

    # Group 3: Historical seed priors
    result["seed_upset_prior_centered"] = pd.Series(
        [
            seed_upset_prior(int(s1), int(s2))
            for s1, s2 in zip(df["Seed1"], df["Seed2"])
        ],
        index=df.index,
    )
    result["seed_x_quality_gap"] = (
        result["seed_upset_prior_centered"] * adjEM_diff
    )

    # Group 4: Quality consistency
    result["quality_source_agreement"] = adjEM_diff * bart_barthag_diff
    result["sos_quality_interaction"] = sos_adjEM_diff * adjEM_diff

    return pd.DataFrame(result, columns=list(MATCHUP_FEATURE_COLUMNS))


def compute_combined_difference_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute KenPom + Barttorvik + matchup interaction features.

    Produces 45 total features: 21 KenPom + 13 Barttorvik + 11 matchup.

    Args:
        df: DataFrame with both KenPom and Barttorvik columns.

    Returns:
        New DataFrame with COMBINED_DIFF_FEATURE_COLUMNS (45 columns).
    """
    kenpom_feats = compute_difference_features(df)
    bart_feats = compute_barttorvik_difference_features(df)
    matchup_feats = compute_matchup_features(df)

    return pd.concat([kenpom_feats, bart_feats, matchup_feats], axis=1)


def symmetrize_training_data(
    X: pd.DataFrame, y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Double training data by swapping team order and flipping labels.

    For difference features, swapping teams negates first-order diffs:
    diff(A,B) = -diff(B,A). Second-order features (products of diffs)
    are even under swap and must be recomputed, not negated.

    Works with both KenPom-only (21 cols) and combined (34 cols) features.

    Args:
        X: Feature matrix with difference columns.
        y: Binary target (1 = Team1 wins).

    Returns:
        Tuple of (augmented X, augmented y) with 2x the original rows.

    Raises:
        ValueError: If X columns don't match expected difference features.
    """
    actual = set(X.columns)
    kenpom_only = set(DIFF_FEATURE_COLUMNS)
    combined = set(COMBINED_DIFF_FEATURE_COLUMNS)

    if actual != kenpom_only and actual != combined:
        raise ValueError(
            f"Symmetrization requires difference features. "
            f"Got {len(actual)} columns, expected {len(kenpom_only)} or {len(combined)}. "
            f"Extra: {actual - combined}, Missing: {combined - actual}"
        )

    # Even-symmetry features: products of two odd terms, or absolute values.
    # These are invariant under team swap and must NOT be negated.
    even_features = {"seed_x_adjEM_interaction"} | MATCHUP_EVEN_FEATURES

    # Build flipped data: negate odd features, preserve even features.
    flipped_data = {}
    for col in X.columns:
        if col in even_features:
            flipped_data[col] = X[col].copy()
        else:
            flipped_data[col] = -X[col]

    # Recompute product features from negated components to ensure
    # numerical consistency: (-a)*(-b) = a*b (same as original).
    flipped_data["seed_x_adjEM_interaction"] = (
        -X["seed_diff"] * -X["adjEM_diff"]
    )

    # Recompute matchup product features if present
    if "offense_defense_product" in X.columns:
        flipped_data["offense_defense_product"] = (
            -X["adjO_diff"] * -X["adjD_diff"]
        )
        flipped_data["bart_offense_defense_product"] = (
            -X["bart_adjOE_diff"] * -X["bart_adjDE_diff"]
        )
        # tempo_mismatch_magnitude is abs() — inherently even, copy is fine
        flipped_data["tempo_x_quality_interaction"] = (
            -X["adjT_diff"] * -X["adjEM_diff"]
        )
        flipped_data["tempo_x_seed_interaction"] = (
            -X["adjT_diff"] * -X["seed_diff"]
        )
        flipped_data["seed_x_quality_gap"] = (
            -X["seed_upset_prior_centered"] * -X["adjEM_diff"]
        )
        flipped_data["quality_source_agreement"] = (
            -X["adjEM_diff"] * -X["bart_barthag_diff"]
        )
        flipped_data["sos_quality_interaction"] = (
            -X["sos_adjEM_diff"] * -X["adjEM_diff"]
        )

    X_flipped = pd.DataFrame(flipped_data, columns=list(X.columns))
    y_flipped = 1 - y

    X_sym = pd.concat([X, X_flipped], ignore_index=True)
    y_sym = pd.concat([y, y_flipped], ignore_index=True)

    return X_sym, y_sym


def load_and_prepare(
    feature_mode: str = "difference",
) -> PreparedData:
    """Load training_data.csv and prepare the feature matrix.

    Args:
        feature_mode: "difference" for KenPom-only diff features (21 cols),
                      "combined" for KenPom + Barttorvik diffs (34 cols),
                      "raw" for original 36 team columns.

    Returns a PreparedData instance with features, target, and metadata.
    """
    df = pd.read_csv(mm_config.MM_TRAINING_DATA)
    logger.info("Loaded %d rows from %s", len(df), mm_config.MM_TRAINING_DATA)

    # Preserve team info before dropping
    info_cols = [c for c in TEAM_INFO_COLUMNS if c in df.columns]
    team_info = df[info_cols].copy()

    years = sorted(df[YEAR_COLUMN].unique().tolist())

    has_bart = _has_barttorvik_columns(df)

    if feature_mode == "combined":
        if not has_bart:
            raise ValueError(
                "feature_mode='combined' requires Barttorvik columns in "
                "training_data.csv. Run the Barttorvik scraper and re-merge."
            )
        X = compute_combined_difference_features(df)
    elif feature_mode == "difference":
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
        feature_mode: "difference" for KenPom-only diff features,
                      "combined" for KenPom + Barttorvik diffs,
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

    has_bart = _has_barttorvik_columns(df)

    if feature_mode == "combined":
        if not has_bart:
            raise ValueError(
                "feature_mode='combined' requires Barttorvik columns in "
                "prediction_data.csv. Run the Barttorvik scraper and re-merge."
            )
        X_pred = compute_combined_difference_features(df)
    elif feature_mode == "difference":
        X_pred = compute_difference_features(df)
    else:
        drop = DROP_COLUMNS + [TARGET_COLUMN]
        X_pred = df.drop(columns=[c for c in drop if c in df.columns])

    return X_pred, team_info
