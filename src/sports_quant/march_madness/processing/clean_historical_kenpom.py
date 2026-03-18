"""Clean and normalize historical KenPom data (2011-2016) into our schema.

Reads pre-tournament KenPom CSVs from two different source schemas
(awaciern/MarchMadness for 2012-2016, christoukmaji for 2011) and
normalizes them to match our existing kenpom_data.csv column format.
"""

import logging
from pathlib import Path

import pandas as pd

from sports_quant.march_madness._config import MM_KENPOM_DIR

logger = logging.getLogger(__name__)

HISTORICAL_DIR = MM_KENPOM_DIR / "historical"
OUTPUT_FILE = MM_KENPOM_DIR / "kenpom_data.csv"

# Our target schema
TARGET_COLUMNS = [
    "Rank", "Team", "Conference", "W-L", "AdjEM",
    "AdjustO", "AdjustO Rank", "AdjustD", "AdjustD Rank",
    "AdjustT", "AdjustT Rank", "Luck", "Luck Rank",
    "SOS AdjEM", "SOS AdjEM Rank", "SOS OppO", "SOS OppO Rank",
    "SOS OppD", "SOS OppD Rank", "NCSOS AdjEM", "NCSOS AdjEM Rank",
    "Year",
]

# Column mapping for awaciern/MarchMadness files (2012-2016)
AWACIERN_MAP = {
    "Rk_AdjEM": "Rank",
    "Team": "Team",
    "Conf": "Conference",
    "W-L": "W-L",
    "AdjEM": "AdjEM",
    "AdjO": "AdjustO",
    "Rk_AdjO": "AdjustO Rank",
    "AdjD": "AdjustD",
    "Rk_AdjD": "AdjustD Rank",
    "AdjT": "AdjustT",
    "Rk_AdjT": "AdjustT Rank",
    "Luck": "Luck",
    "Rk_Luck": "Luck Rank",
    "SOS_AdjEM": "SOS AdjEM",
    "Rk_SOS_AdjEM": "SOS AdjEM Rank",
    "SOS_AdjO": "SOS OppO",
    "Rk_SOS_AdjO": "SOS OppO Rank",
    "SOS_AdjD": "SOS OppD",
    "Rk_SOS_AdjD": "SOS OppD Rank",
    "NCSOS_AdjEM": "NCSOS AdjEM",
    "Rk_NCSOS_AdjEM": "NCSOS AdjEM Rank",
}

# Column mapping for christoukmaji files (2011)
CHRISTOUKMAJI_MAP = {
    "Rank": "Rank",
    "Pure Text": "Team",
    "Conference": "Conference",
    "W-L": "W-L",
    "AdjEM": "AdjEM",
    "AdjO": "AdjustO",
    "R_AdjO": "AdjustO Rank",
    "AdjD": "AdjustD",
    "R_AdjD": "AdjustD Rank",
    "AdjT": "AdjustT",
    "R_AdjT": "AdjustT Rank",
    "Luck": "Luck",
    "R_Luck": "Luck Rank",
    "AdjEM1": "SOS AdjEM",
    "R_AdjEM1": "SOS AdjEM Rank",
    "OppO": "SOS OppO",
    "R_OppO": "SOS OppO Rank",
    "OppD": "SOS OppD",
    "R_OppD": "SOS OppD Rank",
    "AdjEM2": "NCSOS AdjEM",
    "R_AdjEM2": "NCSOS AdjEM Rank",
}


def _clean_awaciern(filepath: Path, year: int) -> pd.DataFrame:
    """Clean a single awaciern-format CSV (2012-2016)."""
    df = pd.read_csv(filepath)
    available = {col: AWACIERN_MAP[col] for col in AWACIERN_MAP if col in df.columns}
    renamed = df[list(available.keys())].rename(columns=available)
    renamed["Year"] = year
    return renamed[TARGET_COLUMNS]


def _clean_christoukmaji(filepath: Path, year: int) -> pd.DataFrame:
    """Clean a single christoukmaji-format CSV (2011)."""
    df = pd.read_csv(filepath)
    available = {
        col: CHRISTOUKMAJI_MAP[col]
        for col in CHRISTOUKMAJI_MAP
        if col in df.columns
    }
    renamed = df[list(available.keys())].rename(columns=available)
    renamed["Year"] = year
    return renamed[TARGET_COLUMNS]


def clean_historical_kenpom() -> pd.DataFrame:
    """Clean all historical KenPom files and return a unified DataFrame.

    Returns:
        DataFrame with TARGET_COLUMNS for years 2011-2016.
    """
    frames: list[pd.DataFrame] = []

    # 2011 (christoukmaji format)
    path_2011 = HISTORICAL_DIR / "kenpom_2011_pretourney.csv"
    if path_2011.exists():
        df = _clean_christoukmaji(path_2011, 2011)
        logger.info("Cleaned 2011: %d rows", len(df))
        frames.append(df)

    # 2012-2016 (awaciern format)
    for year in range(2012, 2017):
        filepath = HISTORICAL_DIR / f"kenpom_{year}_pretourney.csv"
        if filepath.exists():
            df = _clean_awaciern(filepath, year)
            logger.info("Cleaned %d: %d rows", year, len(df))
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No historical KenPom files found in {HISTORICAL_DIR}"
        )

    return pd.concat(frames, ignore_index=True)


def merge_with_existing() -> None:
    """Merge historical data into the existing kenpom_data.csv.

    Appends 2011-2016 rows, removes any duplicate year/team combos,
    and writes back sorted by Year then Rank.
    """
    historical = clean_historical_kenpom()

    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        logger.info(
            "Existing kenpom_data.csv: %d rows, years %s",
            len(existing), sorted(existing["Year"].unique()),
        )

        # Remove any existing rows for years we're adding
        historical_years = set(historical["Year"].unique())
        existing_filtered = existing[~existing["Year"].isin(historical_years)]

        combined = pd.concat(
            [existing_filtered, historical], ignore_index=True,
        )
    else:
        combined = historical

    combined = combined.sort_values(
        ["Year", "Rank"], ascending=[True, True],
    ).reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False)
    logger.info(
        "Wrote %d rows to %s (years: %s)",
        len(combined), OUTPUT_FILE, sorted(combined["Year"].unique()),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    merge_with_existing()
