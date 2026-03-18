"""NCAA tournament schedule scraper for game dates.

Downloads game dates from the ESPN public scoreboard API. Each
tournament day is queried to collect game-level date mappings.

The resulting CSV enables day-slot derivation for the survivor pool
(e.g. R64_D1, R64_D2) since each round spans two calendar days.

ESPN API endpoint (free, unauthenticated):
    https://site.api.espn.com/apis/site/v2/sports/basketball/
    mens-college-basketball/scoreboard?dates=YYYYMMDD&groups=100&limit=50
"""

import json
import logging
import time
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._bracket import ROUND_NAMES
from sports_quant.march_madness._features import standardize_team_name

logger = logging.getLogger(__name__)

_ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
    "?dates={date}&groups=100&limit=50"
)

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# First Thursday of R64 for each tournament year.
# Used to seed the date-range queries.
_R64_START_DATES: dict[int, str] = {
    2008: "2008-03-20",
    2009: "2009-03-19",
    2010: "2010-03-18",
    2011: "2011-03-17",
    2012: "2012-03-15",
    2013: "2013-03-21",
    2014: "2014-03-20",
    2015: "2015-03-19",
    2016: "2016-03-17",
    2017: "2017-03-16",
    2018: "2018-03-15",
    2019: "2019-03-21",
    # 2020: no tournament
    2021: "2021-03-19",
    2022: "2022-03-17",
    2023: "2023-03-16",
    2024: "2024-03-21",
    2025: "2025-03-20",
    2026: "2026-03-19",
}

DEFAULT_YEARS: list[int] = sorted(_R64_START_DATES.keys())

# Number of tournament games per round (excluding First Four)
_ROUND_GAME_COUNTS: dict[int, int] = {
    64: 32, 32: 16, 16: 8, 8: 4, 4: 2, 2: 1,
}

# ESPN team name -> our standardized name overrides.
# Most ESPN names match or are handled by standardize_team_name,
# but some need explicit mapping.
_ESPN_NAME_OVERRIDES: dict[str, str] = {
    "UConn Huskies": "Connecticut",
    "UConn": "Connecticut",
    "Ole Miss Rebels": "Mississippi",
    "Ole Miss": "Mississippi",
    "BYU Cougars": "BYU",
    "VCU Rams": "VCU",
    "LSU Tigers": "LSU",
    "TCU Horned Frogs": "TCU",
    "USC Trojans": "USC",
    "SMU Mustangs": "SMU",
    "UNLV Rebels": "UNLV",
    "UCF Knights": "UCF",
    "UAB Blazers": "UAB",
    "ETSU Buccaneers": "ETSU",
    "SIU-Edwardsville Cougars": "SIUE",
    "SIUE": "SIUE",
    "LIU Sharks": "LIU Brooklyn",
    "LIU Brooklyn Blackbirds": "LIU Brooklyn",
}


def _espn_name_to_standard(espn_name: str) -> str:
    """Convert an ESPN team display name to our standard name.

    Strips common suffixes (mascot names) and applies overrides.
    """
    # Check direct overrides first
    if espn_name in _ESPN_NAME_OVERRIDES:
        return standardize_team_name(_ESPN_NAME_OVERRIDES[espn_name])

    # ESPN names often include the mascot: "Duke Blue Devils"
    # Try the full name through standardize first
    standardized = standardize_team_name(espn_name)
    if standardized != espn_name:
        return standardized

    # No mapping found — return as-is (will be matched during join)
    return espn_name


def _fetch_espn_scoreboard(query_date: str) -> dict:
    """Fetch the ESPN scoreboard JSON for a single date.

    Args:
        query_date: Date string in YYYYMMDD format.

    Returns:
        Parsed JSON dict, or empty dict on failure.
    """
    url = _ESPN_SCOREBOARD_URL.format(date=query_date)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", _REQUEST_HEADERS["User-Agent"])

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
    except Exception as e:
        logger.error("ESPN fetch failed for %s: %s", query_date, e)
        return {}


def _tournament_date_range(year: int) -> list[date]:
    """Generate the list of tournament dates to query for a year.

    Covers First Four (2 days before R64) through NCG (~3 weeks).
    """
    r64_start = _R64_START_DATES.get(year)
    if r64_start is None:
        return []

    start = date.fromisoformat(r64_start) - timedelta(days=2)  # First Four
    end = start + timedelta(days=25)  # NCG is ~3 weeks after FF

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def parse_espn_scoreboard(data: dict, query_date: str) -> list[dict]:
    """Parse game records from ESPN scoreboard JSON.

    Args:
        data: ESPN scoreboard API response.
        query_date: The date queried (YYYY-MM-DD format).

    Returns:
        List of game dicts with YEAR, Team1, Team2, Seed1, Seed2,
        game_date fields.
    """
    events = data.get("events", [])
    rows: list[dict] = []

    for event in events:
        # Extract game date from the event
        game_date_str = event.get("date", "")
        if game_date_str:
            # ESPN dates are UTC (e.g. "2025-03-21T04:30Z").
            # Convert to Eastern so late-night ET games keep the
            # correct calendar date instead of rolling to the next day.
            dt_utc = datetime.fromisoformat(
                game_date_str.replace("Z", "+00:00"),
            )
            dt_eastern = dt_utc.astimezone(ZoneInfo("America/New_York"))
            game_date = dt_eastern.date().isoformat()
        else:
            game_date = query_date

        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue

        teams: list[dict] = []
        for comp in competitors:
            team_info = comp.get("team", {})
            team_name = team_info.get("displayName", "")
            short_name = team_info.get("shortDisplayName", "")
            location = team_info.get("location", "")

            # Use location (school name) when available, fall back to others
            name = location or short_name or team_name

            seed_str = comp.get("curatedRank", {}).get("current", 0)
            if seed_str == 0:
                # Try alternative seed location
                seed_str = comp.get("seed", "0")
            try:
                seed = int(seed_str)
            except (ValueError, TypeError):
                seed = 0

            teams.append({"name": name, "seed": seed})

        if len(teams) == 2:
            year = int(game_date[:4])
            # Tournament year: if March game, it's that year
            # ESPN gives the calendar year
            rows.append({
                "YEAR": year,
                "Team1": _espn_name_to_standard(teams[0]["name"]),
                "Team2": _espn_name_to_standard(teams[1]["name"]),
                "Seed1": teams[0]["seed"],
                "Seed2": teams[1]["seed"],
                "game_date": game_date,
            })

    return rows


def _assign_round_numbers(rows: list[dict]) -> list[dict]:
    """Assign round numbers based on game counts and date ordering.

    The NCAA tournament follows a strict schedule:
    - First Four: 4 games on the earliest dates (may be absent)
    - R64: 32 games across 2-3 days
    - R32: 16 games across 2-3 days
    - S16: 8 games across 2-3 days
    - E8: 4 games across 1-2 days
    - F4: 2 games (1-2 days)
    - NCG: 1 game

    Detects First Four presence by total game count (67 = with FF,
    63 = without). Games are sorted by date, then assigned to rounds
    by cumulative count.
    """
    sorted_rows = sorted(rows, key=lambda r: r["game_date"])
    total = len(sorted_rows)

    # Detect First Four: 67 total = FF present, 63 = FF absent
    has_first_four = total >= 67

    if has_first_four:
        round_targets = [
            (0, 4),    # First Four: round_num=0 (will be filtered)
            (64, 32),  # R64
            (32, 16),  # R32
            (16, 8),   # S16
            (8, 4),    # E8
            (4, 2),    # F4
            (2, 1),    # NCG
        ]
    else:
        round_targets = [
            (64, 32),  # R64
            (32, 16),  # R32
            (16, 8),   # S16
            (8, 4),    # E8
            (4, 2),    # F4
            (2, 1),    # NCG
        ]

    result: list[dict] = []
    round_idx = 0

    for row in sorted_rows:
        if round_idx >= len(round_targets):
            break

        target_round, target_count = round_targets[round_idx]
        result.append({**row, "round_num": target_round})

        current_round_games = sum(
            1 for r in result if r["round_num"] == target_round
        )
        if current_round_games >= target_count:
            round_idx += 1

    # Filter out First Four (round_num=0)
    return [r for r in result if r["round_num"] != 0]


def scrape_schedule_for_year(year: int, rate_limit: float = 1.0) -> pd.DataFrame:
    """Scrape tournament game dates for a single year via ESPN API.

    Queries each tournament date and collects all games.

    Args:
        year: Tournament year.
        rate_limit: Seconds between API requests.

    Returns:
        DataFrame with columns: YEAR, Team1, Team2, Seed1, Seed2,
        round_num, game_date.
    """
    dates = _tournament_date_range(year)
    if not dates:
        logger.warning("No tournament dates configured for year %d", year)
        return pd.DataFrame()

    all_rows: list[dict] = []
    seen_matchups: set[frozenset[str]] = set()

    for i, d in enumerate(dates):
        query = d.strftime("%Y%m%d")
        data = _fetch_espn_scoreboard(query)

        if data:
            day_rows = parse_espn_scoreboard(data, d.isoformat())
            for row in day_rows:
                # Deduplicate by team pair
                key = frozenset({row["Team1"], row["Team2"]})
                if key not in seen_matchups:
                    seen_matchups.add(key)
                    all_rows.append(row)

        if i < len(dates) - 1:
            time.sleep(rate_limit)

    if not all_rows:
        logger.warning("No games found for year %d", year)
        return pd.DataFrame()

    # Assign round numbers based on game count ordering
    all_rows = _assign_round_numbers(all_rows)

    logger.info(
        "Scraped %d tournament games for year %d", len(all_rows), year,
    )
    return pd.DataFrame(all_rows)


def scrape_schedule(
    years: list[int] | None = None,
    output_path: str | None = None,
    rate_limit: float = 1.0,
) -> pd.DataFrame:
    """Scrape tournament game dates for multiple years via ESPN API.

    Args:
        years: List of tournament years. Defaults to DEFAULT_YEARS.
        output_path: Output CSV path. Defaults to MM_SCHEDULE_DATA.
        rate_limit: Seconds between API requests.

    Returns:
        Combined DataFrame of all schedule data.
    """
    years = years or DEFAULT_YEARS
    output_path = output_path or str(mm_config.MM_SCHEDULE_DATA)

    frames: list[pd.DataFrame] = []

    for i, year in enumerate(years):
        logger.info(
            "Scraping schedule for year %d (%d/%d)", year, i + 1, len(years),
        )

        try:
            df = scrape_schedule_for_year(year, rate_limit=rate_limit)
            if not df.empty:
                frames.append(df)
            else:
                logger.warning("No games parsed for year %d", year)
        except Exception as e:
            logger.error("Failed to scrape year %d: %s", year, e)

    if not frames:
        logger.warning("No schedule data scraped.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Save output
    mm_config.MM_SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(
        "Schedule data saved to %s (%d rows, %d years)",
        output_path, len(combined), len(frames),
    )

    return combined


if __name__ == "__main__":
    scrape_schedule()
