"""Barttorvik T-Rank data downloader for March Madness.

Downloads PRE-TOURNAMENT team ratings from the Barttorvik Time Machine.
The Time Machine provides historical daily snapshots, ensuring we get
ratings BEFORE any tournament games are played — preventing data leakage.

IMPORTANT: The season-end CSV at barttorvik.com/{year}_team_results.csv
includes tournament results and MUST NOT be used for tournament prediction.
"""

import gzip
import json
import logging
import urllib.request

import pandas as pd

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._features import standardize_team_name

logger = logging.getLogger(__name__)

# Pre-tournament snapshot dates (day before R64 starts for each year).
# Using the day before R64 captures First Four results in the ratings
# while still excluding all NCAA tournament main-draw games.
# Historical dates (2010-2025) used Selection Sunday and are kept as-is
# since that data was already scraped.
SNAPSHOT_DATES: dict[int, str] = {
    2010: "20100314",
    2011: "20110313",
    2012: "20120311",
    2013: "20130316",
    2014: "20140316",
    2015: "20150315",
    2016: "20160313",
    2017: "20170312",
    2018: "20180311",
    2019: "20190317",
    # 2020: no tournament
    2021: "20210314",
    2022: "20220313",
    2023: "20230312",
    2024: "20240317",
    2025: "20250316",
    2026: "20260318",
}

DEFAULT_YEARS: list[int] = sorted(SNAPSHOT_DATES.keys())

_TIME_MACHINE_URL = (
    "https://barttorvik.com/timemachine/team_results/"
    "{date}_team_results.json.gz"
)

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Column indices in the Time Machine JSON arrays.
# The JSON is a list of lists (no headers), aligned with the CSV columns.
_COL_IDX = {
    "Team": 1,
    "Bart_Rank": 0,
    "Bart_AdjOE": 4,
    "Bart_AdjDE": 6,
    "Bart_Barthag": 8,
    "Bart_AdjT": 44,
    "Bart_SOS": 15,
    "Bart_NCSOS": 16,
    "Bart_EliteSOS": 21,
    "Bart_WAB": 41,
    "Bart_QualO": 29,
    "Bart_QualD": 30,
    "Bart_QualBarthag": 31,
}


def _download_time_machine(year: int) -> pd.DataFrame:
    """Download pre-tournament Barttorvik snapshot from the Time Machine.

    Args:
        year: Season year (e.g. 2025 for the 2024-25 season).

    Returns:
        DataFrame with standardized column names, or empty on error.
    """
    date_str = SNAPSHOT_DATES.get(year)
    if date_str is None:
        logger.warning(
            "No snapshot date configured for year %d", year,
        )
        return pd.DataFrame()

    url = _TIME_MACHINE_URL.format(date=date_str)
    try:
        # Use urllib instead of requests to avoid automatic gzip
        # decompression issues with some corrupted .json.gz files.
        req = urllib.request.Request(url)
        req.add_header("User-Agent", _REQUEST_HEADERS["User-Agent"])
        req.add_header("Accept-Encoding", "identity")

        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        return pd.DataFrame()

    try:
        # Try gzip decompression first, fall back to raw bytes
        try:
            decompressed = gzip.decompress(raw_bytes)
        except (gzip.BadGzipFile, OSError):
            decompressed = raw_bytes
        records = json.loads(decompressed)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("JSON parsing failed for %s: %s", url, e)
        return pd.DataFrame()

    if not isinstance(records, list) or len(records) == 0:
        logger.warning("Empty or invalid data for year %d", year)
        return pd.DataFrame()

    # Extract columns we need from the positional JSON arrays
    rows = []
    for rec in records:
        if not isinstance(rec, list) or len(rec) < 45:
            continue
        row = {}
        for col_name, idx in _COL_IDX.items():
            row[col_name] = rec[idx]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Year"] = year

    # Standardize team names
    df["Team"] = (
        df["Team"]
        .astype(str)
        .str.strip()
        .apply(standardize_team_name)
    )

    # Ensure numeric columns
    numeric_cols = [c for c in df.columns if c not in ("Team", "Year")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "Downloaded %d teams for year %d (snapshot: %s)",
        len(df), year, date_str,
    )
    return df


def scrape_barttorvik(
    years: list[int] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Download pre-tournament Barttorvik T-Rank data from the Time Machine.

    Uses day-before-R64 snapshots to ensure NO tournament main-draw data
    contaminates the ratings while capturing First Four results.

    Args:
        years: List of season years. Defaults to DEFAULT_YEARS.
        output_path: Output CSV path. Defaults to mm_config.MM_BARTTORVIK_DATA.

    Returns:
        Combined DataFrame of all downloaded Barttorvik data.
    """
    years = years or DEFAULT_YEARS
    output_path = output_path or str(mm_config.MM_BARTTORVIK_DATA)

    frames = [_download_time_machine(year) for year in years]
    frames = [f for f in frames if not f.empty]

    if not frames:
        logger.warning("No Barttorvik data downloaded.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Save output
    mm_config.MM_BARTTORVIK_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Barttorvik pre-tournament data saved to %s (%d rows, %d years)",
        output_path, len(df), len(frames),
    )

    return df


if __name__ == "__main__":
    scrape_barttorvik()
