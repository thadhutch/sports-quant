"""Data loading and preparation for March Madness models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._features import (
    DROP_COLUMNS,
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


def load_and_prepare() -> PreparedData:
    """Load training_data.csv and prepare the feature matrix.

    Returns a PreparedData instance with features, target, and metadata.
    """
    df = pd.read_csv(mm_config.MM_TRAINING_DATA)
    logger.info("Loaded %d rows from %s", len(df), mm_config.MM_TRAINING_DATA)

    # Preserve team info before dropping
    info_cols = [c for c in TEAM_INFO_COLUMNS if c in df.columns]
    team_info = df[info_cols].copy()

    # Drop non-feature columns
    df_filtered = df.drop(
        columns=[c for c in DROP_COLUMNS if c in df.columns]
    )

    X = df_filtered.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = df_filtered[TARGET_COLUMN]
    years = sorted(df[YEAR_COLUMN].unique().tolist())

    logger.info("Features: %d columns, years: %s", X.shape[1], years)
    return PreparedData(df=df, X=X, y=y, years=years, team_info=team_info)


def load_prediction_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prediction_data.csv for inference.

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

    # Drop non-feature columns (including Team1_Win)
    drop = DROP_COLUMNS + [TARGET_COLUMN]
    X_pred = df.drop(columns=[c for c in drop if c in df.columns])

    return X_pred, team_info
