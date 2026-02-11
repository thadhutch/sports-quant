"""Data loading and preparation for O/U prediction models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_quant import _config as config
from sports_quant.modeling._features import ALL_FEATURES

logger = logging.getLogger(__name__)

TARGET_COLUMN = "total"
DATE_COLUMN = "Formatted Date"


@dataclass
class PreparedData:
    """Container for model-ready data."""

    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    seasons: np.ndarray
    test_dates: np.ndarray


def load_and_prepare(min_training_seasons: int = 2) -> PreparedData:
    """Load the ranked dataset and prepare it for modeling.

    Steps:
      1. Load OVERUNDER_RANKED CSV.
      2. Parse the date column as datetime and sort by date.
      3. Filter out week-1 games (home_gp == 0 or away_gp == 0).
      4. Drop rows with NaN in any feature or target column.
      5. Compute test dates starting after *min_training_seasons* full seasons.

    Returns a :class:`PreparedData` instance.
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d rows from %s", len(df), config.OVERUNDER_RANKED)

    # Parse dates
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Filter out week-1 games (no prior PFF data)
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    # Drop rows missing any feature or target value
    cols_needed = ALL_FEATURES + [TARGET_COLUMN]
    df = df.dropna(subset=cols_needed)
    logger.info("After filtering: %d games", len(df))

    # Ensure season is numeric
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # Build feature matrix — use only the curated feature list
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    # Determine test dates: start testing after min_training_seasons full seasons
    seasons = sorted(df["season"].dropna().unique())
    logger.info("Unique seasons: %s", seasons)

    if len(seasons) < min_training_seasons + 1:
        raise ValueError(
            f"Need at least {min_training_seasons + 1} seasons, "
            f"but only found {len(seasons)}"
        )

    start_season = seasons[min_training_seasons]
    start_date = df[df["season"] == start_season][DATE_COLUMN].min()
    test_dates = sorted(df[df[DATE_COLUMN] >= start_date][DATE_COLUMN].unique())
    logger.info(
        "Backtesting starts at season %s (%s) — %d test dates",
        start_season,
        start_date.date(),
        len(test_dates),
    )

    return PreparedData(
        df=df,
        X=X,
        y=y,
        seasons=np.array(seasons),
        test_dates=np.array(test_dates),
    )
