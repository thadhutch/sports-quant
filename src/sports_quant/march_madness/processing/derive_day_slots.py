"""Derive survivor pool day slots from tournament schedule data.

Maps each tournament game to a day slot (e.g. R64_D1, R64_D2) based
on the calendar date the game was played. Rounds that span two days
(R64, R32, S16) get split into D1/D2 slots. Later rounds (E8, F4, NCG)
are single slots even if they span multiple days.

Day slots are the assignment targets for the survivor pool optimizer:
you must pick one team per day slot, not one per round.
"""

from __future__ import annotations

import logging

import pandas as pd

from sports_quant.march_madness._bracket import ROUND_NAMES
from sports_quant.march_madness._features import standardize_team_name

logger = logging.getLogger(__name__)

# Survivor pool day slots in pick order
SURVIVOR_SLOTS: tuple[str, ...] = (
    "R64_D1", "R64_D2",
    "R32_D1", "R32_D2",
    "S16_D1", "S16_D2",
    "E8", "F4", "NCG",
)

# Rounds that split into two day-based slots
_TWO_DAY_ROUNDS: frozenset[str] = frozenset({"R64", "R32", "S16"})

# Rounds that are single slots (even if spanning multiple calendar days)
_SINGLE_DAY_ROUNDS: frozenset[str] = frozenset({"E8", "F4", "NCG"})

# Round number -> round name (from _bracket.py)
_ROUND_NUM_TO_NAME: dict[int, str] = dict(ROUND_NAMES)


def derive_day_slots(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Derive day_slot column from schedule data.

    For each (year, round), sorts unique game dates chronologically.
    The earliest date becomes D1, the second becomes D2 for rounds
    that span two days. Single-day rounds keep their round name as-is.

    Args:
        schedule_df: DataFrame with columns YEAR, round_num, game_date.
            round_num uses the bracket convention (64, 32, 16, 8, 4, 2).

    Returns:
        New DataFrame with an added ``day_slot`` column.

    Raises:
        ValueError: If a two-day round has fewer than 2 unique dates.
    """
    if schedule_df.empty:
        return schedule_df.assign(day_slot=pd.Series(dtype=str))

    result_rows: list[dict] = []

    for year, year_group in schedule_df.groupby("YEAR"):
        for round_num, round_group in year_group.groupby("round_num"):
            round_name = _ROUND_NUM_TO_NAME.get(int(round_num))
            if round_name is None:
                logger.warning(
                    "Unknown round_num %d in year %d, skipping",
                    round_num, year,
                )
                continue

            unique_dates = sorted(round_group["game_date"].unique())

            if round_name in _TWO_DAY_ROUNDS:
                if len(unique_dates) < 2:
                    logger.warning(
                        "Round %s in %d has only %d unique date(s): %s. "
                        "Assigning all games to D1.",
                        round_name, year, len(unique_dates), unique_dates,
                    )
                    for _, row in round_group.iterrows():
                        result_rows.append({
                            **row.to_dict(),
                            "day_slot": f"{round_name}_D1",
                        })
                    continue

                day1_date = unique_dates[0]
                day2_date = unique_dates[1]

                if len(unique_dates) > 2:
                    logger.warning(
                        "Round %s in %d has %d dates: %s. "
                        "Using first two as D1/D2.",
                        round_name, year, len(unique_dates), unique_dates,
                    )

                for _, row in round_group.iterrows():
                    if row["game_date"] == day1_date:
                        slot = f"{round_name}_D1"
                    elif row["game_date"] == day2_date:
                        slot = f"{round_name}_D2"
                    else:
                        # Games on 3rd+ date go to D2
                        slot = f"{round_name}_D2"

                    result_rows.append({**row.to_dict(), "day_slot": slot})

            elif round_name in _SINGLE_DAY_ROUNDS:
                for _, row in round_group.iterrows():
                    result_rows.append({
                        **row.to_dict(),
                        "day_slot": round_name,
                    })
            else:
                logger.warning("Unhandled round %s", round_name)

    return pd.DataFrame(result_rows)


def validate_day_slots(df: pd.DataFrame) -> dict[int, list[str]]:
    """Validate that each year produces exactly 9 survivor slots.

    Args:
        df: DataFrame with YEAR and day_slot columns.

    Returns:
        Dict of year -> list of issues. Empty list means valid.
    """
    issues: dict[int, list[str]] = {}

    for year, group in df.groupby("YEAR"):
        year_issues: list[str] = []
        slots_present = set(group["day_slot"].unique())

        missing = set(SURVIVOR_SLOTS) - slots_present
        extra = slots_present - set(SURVIVOR_SLOTS)

        if missing:
            year_issues.append(f"Missing slots: {sorted(missing)}")
        if extra:
            year_issues.append(f"Extra slots: {sorted(extra)}")

        # Check game counts per slot (wide tolerances — real schedules
        # can be uneven when games bleed into a 3rd calendar day)
        for slot in SURVIVOR_SLOTS:
            slot_games = group[group["day_slot"] == slot]
            count = len(slot_games)
            if slot.startswith("R64"):
                if count < 8 or count > 24:
                    year_issues.append(
                        f"{slot}: {count} games (expected 8-24)"
                    )
            elif slot.startswith("R32"):
                if count < 4 or count > 12:
                    year_issues.append(
                        f"{slot}: {count} games (expected 4-12)"
                    )
            elif slot.startswith("S16"):
                if count < 1 or count > 7:
                    year_issues.append(
                        f"{slot}: {count} games (expected 1-7)"
                    )

        if year_issues:
            issues[year] = year_issues

    return issues


def join_day_slots_to_matchups(
    matchups_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join day_slot onto restructured matchups by team name matching.

    Matches games between the schedule (with day_slot) and matchups
    DataFrames using standardized team names and year.

    Args:
        matchups_df: The restructured_matchups DataFrame.
        schedule_df: Schedule DataFrame with day_slot column.

    Returns:
        New matchups DataFrame with game_date and day_slot columns added.
        Unmatched games get None for both columns.
    """
    if schedule_df.empty or matchups_df.empty:
        return matchups_df.assign(
            game_date=None,
            day_slot=None,
        )

    # Build lookup: (year, frozenset({team1, team2})) -> (game_date, day_slot)
    schedule_lookup: dict[tuple[int, frozenset[str]], tuple[str, str]] = {}

    for _, row in schedule_df.iterrows():
        t1 = standardize_team_name(str(row["Team1"]).strip())
        t2 = standardize_team_name(str(row["Team2"]).strip())
        key = (int(row["YEAR"]), frozenset({t1, t2}))
        schedule_lookup[key] = (row["game_date"], row["day_slot"])

    # Join onto matchups
    game_dates: list[str | None] = []
    day_slots: list[str | None] = []
    unmatched = 0

    for _, row in matchups_df.iterrows():
        t1 = standardize_team_name(str(row["Team1"]).strip())
        t2 = standardize_team_name(str(row["Team2"]).strip())
        key = (int(row["YEAR"]), frozenset({t1, t2}))

        match = schedule_lookup.get(key)
        if match is not None:
            game_dates.append(match[0])
            day_slots.append(match[1])
        else:
            game_dates.append(None)
            day_slots.append(None)
            unmatched += 1

    if unmatched > 0:
        logger.warning(
            "%d of %d matchups could not be matched to schedule data",
            unmatched, len(matchups_df),
        )

    result = matchups_df.copy()
    result["game_date"] = game_dates
    result["day_slot"] = day_slots
    return result
