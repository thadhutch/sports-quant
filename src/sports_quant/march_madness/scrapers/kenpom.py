"""KenPom ratings scraper for March Madness data."""

import logging
import re
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._features import KENPOM_COLUMNS

logger = logging.getLogger(__name__)

# Default Web Archive URLs for historical KenPom data
DEFAULT_URLS: list[str] = [
    "https://web.archive.org/web/20170312131016/http://kenpom.com/",
    "https://web.archive.org/web/20180311122559/https://kenpom.com/",
    "https://web.archive.org/web/20190317211809/https://kenpom.com/",
    "https://web.archive.org/web/20210318152437/https://kenpom.com/",
    "https://web.archive.org/web/20220312213724/https://kenpom.com/",
    "https://web.archive.org/web/20230314165625/https://kenpom.com/",
    "https://web.archive.org/web/20240321081134/https://kenpom.com/",
    "https://web.archive.org/web/20250314000625/https://kenpom.com/",
    "https://kenpom.com/",
]

DEFAULT_YEARS: list[int] = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]


def _scrape_archive(url: str, year: int) -> pd.DataFrame:
    """Fetch and parse a single KenPom page into a DataFrame.

    Args:
        url: KenPom URL (live or web archive).
        year: Season year to tag the data with.

    Returns:
        DataFrame of KenPom ratings for that year, or empty DataFrame on error.
    """
    if year == 2020:
        logger.info("Skipping year 2020 (no tournament data).")
        return pd.DataFrame()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    try:
        page = requests.get(url, headers=headers)
        page.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Request failed for %s: %s", url, e)
        return pd.DataFrame()

    try:
        soup = BeautifulSoup(page.text, features="lxml")
        table_full = soup.find_all("table", {"id": "ratings-table"})

        if not table_full:
            logger.warning("No ratings table found at %s", url)
            return pd.DataFrame()

        thead = table_full[0].find_all("thead")
        table = table_full[0]

        for weird in thead:
            table = str(table).replace(str(weird), "")

        df = pd.read_html(StringIO(table))[0]
    except Exception as e:
        logger.error("HTML processing failed for %s: %s", url, e)
        return pd.DataFrame()

    df["year"] = year
    return df


def scrape_kenpom(
    years: list[int] | None = None,
    urls: list[str] | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Scrape KenPom ratings for the given years and save to CSV.

    Args:
        years: List of season years to scrape. Defaults to DEFAULT_YEARS.
        urls: Corresponding URLs for each year. Defaults to DEFAULT_URLS.
        output_path: Output CSV path. Defaults to mm_config.MM_KENPOM_RAW.

    Returns:
        Combined DataFrame of all scraped KenPom data.
    """
    years = years or DEFAULT_YEARS
    urls = urls or DEFAULT_URLS
    output_path = output_path or mm_config.MM_KENPOM_RAW

    df = pd.DataFrame()
    for year, url in zip(years, urls):
        logger.info("Scraping: %s (year %d)", url, year)
        archive = _scrape_archive(url, year)
        df = pd.concat((df, archive), axis=0)

    if df.empty:
        logger.warning("No data scraped.")
        return df

    df.columns = KENPOM_COLUMNS

    # Clean team names: remove digits and semicolons
    df["Team"] = df["Team"].apply(
        lambda x: re.sub(r"\d", "", x).strip().replace(";", "")
    )

    # Save output
    mm_config.MM_KENPOM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("KenPom data saved to %s (%d rows)", output_path, len(df))

    return df


if __name__ == "__main__":
    scrape_kenpom()
