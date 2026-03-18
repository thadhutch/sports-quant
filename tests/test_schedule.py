"""Tests for schedule scraper and day slot derivation.

Covers ESPN API parsing, round number assignment, day slot
derivation, matchup joining, and validation of 9 survivor slots.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sports_quant.march_madness._bracket import SURVIVOR_SLOTS
from sports_quant.march_madness.processing.derive_day_slots import (
    SURVIVOR_SLOTS as DS_SURVIVOR_SLOTS,
    derive_day_slots,
    join_day_slots_to_matchups,
    validate_day_slots,
)
from sports_quant.march_madness.scrapers.schedule import (
    _assign_round_numbers,
    _espn_name_to_standard,
    parse_espn_scoreboard,
)


# ---------------------------------------------------------------------------
# ESPN name conversion
# ---------------------------------------------------------------------------


class TestEspnNameConversion:
    def test_known_override(self):
        assert _espn_name_to_standard("UConn") == "Connecticut"
        assert _espn_name_to_standard("Ole Miss") == "Mississippi"
        assert _espn_name_to_standard("BYU Cougars") == "BYU"

    def test_passthrough(self):
        """Names without overrides pass through standardize_team_name."""
        result = _espn_name_to_standard("Duke")
        assert result == "Duke"

    def test_standardize_applied(self):
        """standardize_team_name is applied after override check."""
        # "Brigham Young" maps via standardize_team_name
        result = _espn_name_to_standard("Brigham Young")
        assert result == "BYU"


# ---------------------------------------------------------------------------
# ESPN scoreboard parsing
# ---------------------------------------------------------------------------


class TestParseEspnScoreboard:
    """Test ESPN JSON parsing with fixture data."""

    FIXTURE_DATA = {
        "events": [
            {
                "date": "2025-03-20T16:00Z",
                "competitions": [{
                    "competitors": [
                        {
                            "team": {
                                "displayName": "Duke Blue Devils",
                                "shortDisplayName": "Duke",
                                "location": "Duke",
                            },
                            "curatedRank": {"current": 1},
                        },
                        {
                            "team": {
                                "displayName": "Kentucky Wildcats",
                                "shortDisplayName": "Kentucky",
                                "location": "Kentucky",
                            },
                            "curatedRank": {"current": 16},
                        },
                    ],
                }],
            },
            {
                "date": "2025-03-20T19:00Z",
                "competitions": [{
                    "competitors": [
                        {
                            "team": {
                                "displayName": "Kansas Jayhawks",
                                "shortDisplayName": "Kansas",
                                "location": "Kansas",
                            },
                            "curatedRank": {"current": 2},
                        },
                        {
                            "team": {
                                "displayName": "UCLA Bruins",
                                "shortDisplayName": "UCLA",
                                "location": "UCLA",
                            },
                            "curatedRank": {"current": 15},
                        },
                    ],
                }],
            },
        ],
    }

    def test_parses_two_games(self):
        rows = parse_espn_scoreboard(self.FIXTURE_DATA, "2025-03-20")
        assert len(rows) == 2

    def test_extracts_teams(self):
        rows = parse_espn_scoreboard(self.FIXTURE_DATA, "2025-03-20")
        assert rows[0]["Team1"] == "Duke"
        assert rows[0]["Team2"] == "Kentucky"

    def test_extracts_seeds(self):
        rows = parse_espn_scoreboard(self.FIXTURE_DATA, "2025-03-20")
        assert rows[0]["Seed1"] == 1
        assert rows[0]["Seed2"] == 16

    def test_extracts_date(self):
        rows = parse_espn_scoreboard(self.FIXTURE_DATA, "2025-03-20")
        assert rows[0]["game_date"] == "2025-03-20"

    def test_empty_events(self):
        rows = parse_espn_scoreboard({"events": []}, "2025-03-20")
        assert rows == []

    def test_empty_data(self):
        rows = parse_espn_scoreboard({}, "2025-03-20")
        assert rows == []

    def test_utc_midnight_rollover_converts_to_eastern(self):
        """A game at 11:30 PM EDT (03:30 UTC next day) keeps the ET date."""
        data = {
            "events": [
                {
                    "date": "2025-03-21T03:30Z",
                    "competitions": [{
                        "competitors": [
                            {
                                "team": {
                                    "displayName": "Duke Blue Devils",
                                    "shortDisplayName": "Duke",
                                    "location": "Duke",
                                },
                                "curatedRank": {"current": 1},
                            },
                            {
                                "team": {
                                    "displayName": "Kentucky Wildcats",
                                    "shortDisplayName": "Kentucky",
                                    "location": "Kentucky",
                                },
                                "curatedRank": {"current": 16},
                            },
                        ],
                    }],
                },
            ],
        }
        rows = parse_espn_scoreboard(data, "2025-03-20")
        assert rows[0]["game_date"] == "2025-03-20"
        assert rows[0]["YEAR"] == 2025


# ---------------------------------------------------------------------------
# Round number assignment
# ---------------------------------------------------------------------------


class TestAssignRoundNumbers:
    def _make_rows(self, date_counts: list[tuple[str, int]]) -> list[dict]:
        """Create fake game rows with given date/count pairs."""
        rows = []
        game_id = 0
        for date, count in date_counts:
            for _ in range(count):
                rows.append({
                    "YEAR": 2025,
                    "Team1": f"Team{game_id}A",
                    "Team2": f"Team{game_id}B",
                    "Seed1": 1,
                    "Seed2": 16,
                    "game_date": date,
                })
                game_id += 1
        return rows

    def test_standard_tournament(self):
        """Standard 67-game tournament assigns correct rounds."""
        rows = self._make_rows([
            ("2025-03-18", 2),   # First Four day 1
            ("2025-03-19", 2),   # First Four day 2
            ("2025-03-20", 16),  # R64 day 1
            ("2025-03-21", 16),  # R64 day 2
            ("2025-03-22", 8),   # R32 day 1
            ("2025-03-23", 8),   # R32 day 2
            ("2025-03-27", 4),   # S16 day 1
            ("2025-03-28", 4),   # S16 day 2
            ("2025-03-29", 2),   # E8 day 1
            ("2025-03-30", 2),   # E8 day 2
            ("2025-04-05", 2),   # F4
            ("2025-04-07", 1),   # NCG
        ])

        result = _assign_round_numbers(rows)

        # First Four should be filtered out
        assert all(r["round_num"] != 0 for r in result)

        # Check counts per round
        r64 = [r for r in result if r["round_num"] == 64]
        r32 = [r for r in result if r["round_num"] == 32]
        s16 = [r for r in result if r["round_num"] == 16]
        e8 = [r for r in result if r["round_num"] == 8]
        f4 = [r for r in result if r["round_num"] == 4]
        ncg = [r for r in result if r["round_num"] == 2]

        assert len(r64) == 32
        assert len(r32) == 16
        assert len(s16) == 8
        assert len(e8) == 4
        assert len(f4) == 2
        assert len(ncg) == 1

    def test_no_first_four(self):
        """63-game tournament (no First Four) assigns correct rounds."""
        rows = self._make_rows([
            ("2008-03-20", 16),
            ("2008-03-21", 16),
            ("2008-03-22", 8),
            ("2008-03-23", 8),
            ("2008-03-27", 4),
            ("2008-03-28", 4),
            ("2008-03-29", 2),
            ("2008-03-30", 2),
            ("2008-04-05", 2),
            ("2008-04-07", 1),
        ])

        result = _assign_round_numbers(rows)

        # 63 total = no FF detected, all assigned to proper rounds
        assert len(result) == 63
        r64 = [r for r in result if r["round_num"] == 64]
        r32 = [r for r in result if r["round_num"] == 32]
        s16 = [r for r in result if r["round_num"] == 16]
        assert len(r64) == 32
        assert len(r32) == 16
        assert len(s16) == 8


# ---------------------------------------------------------------------------
# Day slot derivation
# ---------------------------------------------------------------------------


class TestDeriveDaySlots:
    def _make_schedule_df(self) -> pd.DataFrame:
        """Create a minimal schedule DataFrame for testing."""
        rows = []

        # R64: 16 on day 1, 16 on day 2
        for i in range(16):
            rows.append({
                "YEAR": 2025, "Team1": f"R64D1_{i}A", "Team2": f"R64D1_{i}B",
                "Seed1": 1, "Seed2": 16, "round_num": 64,
                "game_date": "2025-03-20",
            })
        for i in range(16):
            rows.append({
                "YEAR": 2025, "Team1": f"R64D2_{i}A", "Team2": f"R64D2_{i}B",
                "Seed1": 1, "Seed2": 16, "round_num": 64,
                "game_date": "2025-03-21",
            })

        # R32: 8 on day 1, 8 on day 2
        for i in range(8):
            rows.append({
                "YEAR": 2025, "Team1": f"R32D1_{i}A", "Team2": f"R32D1_{i}B",
                "Seed1": 1, "Seed2": 8, "round_num": 32,
                "game_date": "2025-03-22",
            })
        for i in range(8):
            rows.append({
                "YEAR": 2025, "Team1": f"R32D2_{i}A", "Team2": f"R32D2_{i}B",
                "Seed1": 1, "Seed2": 8, "round_num": 32,
                "game_date": "2025-03-23",
            })

        # S16: 4 on day 1, 4 on day 2
        for i in range(4):
            rows.append({
                "YEAR": 2025, "Team1": f"S16D1_{i}A", "Team2": f"S16D1_{i}B",
                "Seed1": 1, "Seed2": 4, "round_num": 16,
                "game_date": "2025-03-27",
            })
        for i in range(4):
            rows.append({
                "YEAR": 2025, "Team1": f"S16D2_{i}A", "Team2": f"S16D2_{i}B",
                "Seed1": 1, "Seed2": 4, "round_num": 16,
                "game_date": "2025-03-28",
            })

        # E8: 4 games on 2 days (but single slot)
        for i in range(2):
            rows.append({
                "YEAR": 2025, "Team1": f"E8_{i}A", "Team2": f"E8_{i}B",
                "Seed1": 1, "Seed2": 2, "round_num": 8,
                "game_date": "2025-03-29",
            })
        for i in range(2):
            rows.append({
                "YEAR": 2025, "Team1": f"E8_{i+2}A", "Team2": f"E8_{i+2}B",
                "Seed1": 1, "Seed2": 2, "round_num": 8,
                "game_date": "2025-03-30",
            })

        # F4: 2 games
        for i in range(2):
            rows.append({
                "YEAR": 2025, "Team1": f"F4_{i}A", "Team2": f"F4_{i}B",
                "Seed1": 1, "Seed2": 1, "round_num": 4,
                "game_date": "2025-04-05",
            })

        # NCG: 1 game
        rows.append({
            "YEAR": 2025, "Team1": "NCG_A", "Team2": "NCG_B",
            "Seed1": 1, "Seed2": 1, "round_num": 2,
            "game_date": "2025-04-07",
        })

        return pd.DataFrame(rows)

    def test_produces_9_slots(self):
        """Day slot derivation produces all 9 survivor slots."""
        df = self._make_schedule_df()
        result = derive_day_slots(df)

        slots = set(result["day_slot"].unique())
        assert slots == set(SURVIVOR_SLOTS)

    def test_r64_split(self):
        """R64 games are split by date into D1 and D2."""
        df = self._make_schedule_df()
        result = derive_day_slots(df)

        r64_d1 = result[result["day_slot"] == "R64_D1"]
        r64_d2 = result[result["day_slot"] == "R64_D2"]

        assert len(r64_d1) == 16
        assert len(r64_d2) == 16
        assert all(r64_d1["game_date"] == "2025-03-20")
        assert all(r64_d2["game_date"] == "2025-03-21")

    def test_e8_single_slot(self):
        """E8 games get a single slot even across two dates."""
        df = self._make_schedule_df()
        result = derive_day_slots(df)

        e8 = result[result["day_slot"] == "E8"]
        assert len(e8) == 4

    def test_f4_single_slot(self):
        """F4 games get a single slot."""
        df = self._make_schedule_df()
        result = derive_day_slots(df)

        f4 = result[result["day_slot"] == "F4"]
        assert len(f4) == 2

    def test_ncg_single_slot(self):
        """NCG is a single-game slot."""
        df = self._make_schedule_df()
        result = derive_day_slots(df)

        ncg = result[result["day_slot"] == "NCG"]
        assert len(ncg) == 1

    def test_empty_input(self):
        """Empty DataFrame returns empty with day_slot column."""
        df = pd.DataFrame()
        result = derive_day_slots(df)
        assert "day_slot" in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Day slot validation
# ---------------------------------------------------------------------------


class TestValidateDaySlots:
    def test_valid_year(self):
        """Valid data produces no issues."""
        rows = []
        slot_counts = {
            "R64_D1": 16, "R64_D2": 16,
            "R32_D1": 8, "R32_D2": 8,
            "S16_D1": 4, "S16_D2": 4,
            "E8": 4, "F4": 2, "NCG": 1,
        }
        for slot, count in slot_counts.items():
            for i in range(count):
                rows.append({"YEAR": 2025, "day_slot": slot})

        df = pd.DataFrame(rows)
        issues = validate_day_slots(df)
        assert 2025 not in issues or issues[2025] == []

    def test_missing_slot_detected(self):
        """Missing slots are flagged."""
        rows = [{"YEAR": 2025, "day_slot": "R64_D1"}] * 16
        df = pd.DataFrame(rows)
        issues = validate_day_slots(df)
        assert 2025 in issues
        assert any("Missing slots" in msg for msg in issues[2025])


# ---------------------------------------------------------------------------
# Matchup joining
# ---------------------------------------------------------------------------


class TestJoinDaySlotsToMatchups:
    def test_basic_join(self):
        """Day slots are joined to matchups by team names."""
        matchups = pd.DataFrame([
            {"YEAR": 2025, "Team1": "Duke", "Team2": "Kentucky"},
            {"YEAR": 2025, "Team1": "Kansas", "Team2": "UCLA"},
        ])
        schedule = pd.DataFrame([
            {
                "YEAR": 2025, "Team1": "Duke", "Team2": "Kentucky",
                "game_date": "2025-03-20", "day_slot": "R64_D1",
            },
            {
                "YEAR": 2025, "Team1": "Kansas", "Team2": "UCLA",
                "game_date": "2025-03-21", "day_slot": "R64_D2",
            },
        ])

        result = join_day_slots_to_matchups(matchups, schedule)

        assert result.iloc[0]["day_slot"] == "R64_D1"
        assert result.iloc[0]["game_date"] == "2025-03-20"
        assert result.iloc[1]["day_slot"] == "R64_D2"

    def test_team_order_independent(self):
        """Join works regardless of Team1/Team2 ordering."""
        matchups = pd.DataFrame([
            {"YEAR": 2025, "Team1": "Duke", "Team2": "Kentucky"},
        ])
        schedule = pd.DataFrame([
            {
                "YEAR": 2025, "Team1": "Kentucky", "Team2": "Duke",
                "game_date": "2025-03-20", "day_slot": "R64_D1",
            },
        ])

        result = join_day_slots_to_matchups(matchups, schedule)
        assert result.iloc[0]["day_slot"] == "R64_D1"

    def test_unmatched_games_get_none(self):
        """Unmatched matchups get None for day_slot."""
        matchups = pd.DataFrame([
            {"YEAR": 2025, "Team1": "Duke", "Team2": "Kentucky"},
        ])
        schedule = pd.DataFrame(columns=["YEAR", "Team1", "Team2", "game_date", "day_slot"])

        result = join_day_slots_to_matchups(matchups, schedule)
        assert result.iloc[0]["day_slot"] is None
        assert result.iloc[0]["game_date"] is None

    def test_empty_inputs(self):
        """Empty inputs produce empty output with new columns."""
        matchups = pd.DataFrame(columns=["YEAR", "Team1", "Team2"])
        schedule = pd.DataFrame(columns=["YEAR", "Team1", "Team2", "game_date", "day_slot"])

        result = join_day_slots_to_matchups(matchups, schedule)
        assert "day_slot" in result.columns
        assert "game_date" in result.columns


# ---------------------------------------------------------------------------
# SURVIVOR_SLOTS consistency
# ---------------------------------------------------------------------------


class TestSurvivorSlotsConsistency:
    def test_bracket_and_derive_match(self):
        """SURVIVOR_SLOTS in _bracket.py matches derive_day_slots.py."""
        assert SURVIVOR_SLOTS == DS_SURVIVOR_SLOTS

    def test_exactly_9_slots(self):
        assert len(SURVIVOR_SLOTS) == 9

    def test_slot_names(self):
        expected = {
            "R64_D1", "R64_D2", "R32_D1", "R32_D2",
            "S16_D1", "S16_D2", "E8", "F4", "NCG",
        }
        assert set(SURVIVOR_SLOTS) == expected
