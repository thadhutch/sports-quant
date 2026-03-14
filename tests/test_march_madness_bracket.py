"""Tests for March Madness bracket data structures and builder."""

import pandas as pd
import pytest

from sports_quant.march_madness._bracket import (
    ROUND_NAMES,
    ROUND_ORDER,
    Bracket,
    BracketGame,
    BracketSlot,
    determine_upset,
)
from sports_quant.march_madness._bracket_builder import (
    build_actual_bracket,
    build_predicted_bracket,
    compare_brackets,
)


# ---------------------------------------------------------------------------
# BracketSlot
# ---------------------------------------------------------------------------

class TestBracketSlot:
    def test_creation(self):
        slot = BracketSlot(team="Duke", seed=1)
        assert slot.team == "Duke"
        assert slot.seed == 1

    def test_immutability(self):
        slot = BracketSlot(team="Duke", seed=1)
        with pytest.raises(AttributeError):
            slot.team = "UNC"

    def test_display(self):
        slot = BracketSlot(team="Duke", seed=1)
        assert str(slot) == "(1) Duke"


# ---------------------------------------------------------------------------
# BracketGame
# ---------------------------------------------------------------------------

class TestBracketGame:
    def test_creation(self):
        game = BracketGame(
            round_name="R64",
            region=0,
            game_index=0,
            team1=BracketSlot("Duke", 1),
            team2=BracketSlot("FDU", 16),
            winner=BracketSlot("Duke", 1),
            win_probability=0.95,
            is_upset=False,
            is_correct=None,
        )
        assert game.round_name == "R64"
        assert game.winner.team == "Duke"

    def test_immutability(self):
        game = BracketGame(
            round_name="R64",
            region=0,
            game_index=0,
            team1=BracketSlot("Duke", 1),
            team2=BracketSlot("FDU", 16),
            winner=BracketSlot("Duke", 1),
            win_probability=0.95,
            is_upset=False,
            is_correct=None,
        )
        with pytest.raises(AttributeError):
            game.winner = BracketSlot("FDU", 16)

    def test_no_winner(self):
        """Games without a determined winner (future games)."""
        game = BracketGame(
            round_name="R64",
            region=0,
            game_index=0,
            team1=BracketSlot("Duke", 1),
            team2=BracketSlot("FDU", 16),
            winner=None,
            win_probability=None,
            is_upset=False,
            is_correct=None,
        )
        assert game.winner is None


# ---------------------------------------------------------------------------
# determine_upset
# ---------------------------------------------------------------------------

class TestDetermineUpset:
    def test_higher_seed_wins_not_upset(self):
        assert determine_upset(seed1=1, seed2=16, team1_wins=True) is False

    def test_lower_seed_wins_is_upset(self):
        assert determine_upset(seed1=1, seed2=16, team1_wins=False) is True

    def test_equal_seeds_not_upset(self):
        assert determine_upset(seed1=1, seed2=1, team1_wins=True) is False

    def test_team2_is_higher_seed_and_wins_not_upset(self):
        """Team2 has better (lower) seed and wins — not an upset."""
        assert determine_upset(seed1=9, seed2=1, team1_wins=False) is False

    def test_team1_is_lower_seed_and_wins_is_upset(self):
        """Team1 has worse (higher) seed and wins — upset."""
        assert determine_upset(seed1=14, seed2=3, team1_wins=True) is True


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------

class TestBracket:
    def _make_games(self, n=4):
        """Helper to create a tuple of BracketGames."""
        games = []
        for i in range(n):
            games.append(BracketGame(
                round_name="R64",
                region=0,
                game_index=i,
                team1=BracketSlot(f"Team{i*2}", i + 1),
                team2=BracketSlot(f"Team{i*2+1}", 17 - i),
                winner=BracketSlot(f"Team{i*2}", i + 1),
                win_probability=0.8,
                is_upset=False,
                is_correct=None,
            ))
        return tuple(games)

    def test_creation(self):
        games = self._make_games()
        bracket = Bracket(year=2024, source="actual", games=games)
        assert bracket.year == 2024
        assert len(bracket.games) == 4

    def test_immutability(self):
        games = self._make_games()
        bracket = Bracket(year=2024, source="actual", games=games)
        with pytest.raises(AttributeError):
            bracket.year = 2025

    def test_games_by_round(self):
        games = self._make_games()
        bracket = Bracket(year=2024, source="actual", games=games)
        by_round = bracket.games_by_round()
        assert "R64" in by_round
        assert len(by_round["R64"]) == 4

    def test_games_by_region(self):
        games = self._make_games()
        bracket = Bracket(year=2024, source="actual", games=games)
        by_region = bracket.games_by_region()
        assert 0 in by_region
        assert len(by_region[0]) == 4

    def test_accuracy(self):
        """Accuracy counts only games where is_correct is not None."""
        games = (
            BracketGame("R64", 0, 0, BracketSlot("A", 1), BracketSlot("B", 16),
                        BracketSlot("A", 1), 0.9, False, True),
            BracketGame("R64", 0, 1, BracketSlot("C", 2), BracketSlot("D", 15),
                        BracketSlot("C", 2), 0.8, False, False),
            BracketGame("R64", 0, 2, BracketSlot("E", 3), BracketSlot("F", 14),
                        BracketSlot("E", 3), 0.7, False, None),
        )
        bracket = Bracket(year=2024, source="predicted", games=games)
        assert bracket.accuracy() == 0.5  # 1 correct out of 2 evaluated

    def test_accuracy_by_round(self):
        games = (
            BracketGame("R64", 0, 0, BracketSlot("A", 1), BracketSlot("B", 16),
                        BracketSlot("A", 1), 0.9, False, True),
            BracketGame("R64", 0, 1, BracketSlot("C", 2), BracketSlot("D", 15),
                        BracketSlot("C", 2), 0.8, False, False),
            BracketGame("R32", 0, 0, BracketSlot("A", 1), BracketSlot("C", 2),
                        BracketSlot("A", 1), 0.75, False, True),
        )
        bracket = Bracket(year=2024, source="predicted", games=games)
        acc = bracket.accuracy_by_round()
        assert acc["R64"] == 0.5
        assert acc["R32"] == 1.0

    def test_upset_count(self):
        games = (
            BracketGame("R64", 0, 0, BracketSlot("A", 1), BracketSlot("B", 16),
                        BracketSlot("A", 1), 0.9, False, None),
            BracketGame("R64", 0, 1, BracketSlot("C", 2), BracketSlot("D", 15),
                        BracketSlot("D", 15), 0.6, True, None),
        )
        bracket = Bracket(year=2024, source="actual", games=games)
        assert bracket.upset_count() == 1


# ---------------------------------------------------------------------------
# Round constants
# ---------------------------------------------------------------------------

class TestRoundConstants:
    def test_round_names_map_ints(self):
        assert ROUND_NAMES[64] == "R64"
        assert ROUND_NAMES[2] == "NCG"

    def test_round_order_sorted(self):
        """ROUND_ORDER should go from R64 to NCG."""
        assert ROUND_ORDER == ("R64", "R32", "S16", "E8", "F4", "NCG")


# ---------------------------------------------------------------------------
# build_actual_bracket
# ---------------------------------------------------------------------------

def _make_matchups_df(year=2024):
    """Create a minimal restructured_matchups DataFrame for one year.

    Builds 8 R64 games (2 per region) + 4 R32 games + 2 S16 games
    + 1 E8 + 1 F4 + 1 NCG = 17 games total (not a full 63, but enough
    to verify the builder logic).
    """
    rows = []
    seeds_r64 = [(1, 16), (8, 9), (1, 16), (8, 9),
                 (1, 16), (8, 9), (1, 16), (8, 9)]
    teams_r64 = [
        ("Duke", "FDU"), ("UCLA", "Boise"),
        ("UConn", "Wagner"), ("Iowa St", "SDSU"),
        ("Houston", "Longwood"), ("Purdue", "Grambling"),
        ("Auburn", "ALST"), ("Kansas", "Howard"),
    ]
    for i, ((t1, t2), (s1, s2)) in enumerate(zip(teams_r64, seeds_r64)):
        rows.append({
            "YEAR": year, "Team1": t1, "Seed1": s1,
            "Team2": t2, "Seed2": s2, "CURRENT ROUND": 64,
            "Score1": 80.0, "Score2": 60.0, "Team1_Win": 1,
        })

    # R32: winners from R64 pairs
    r32_games = [
        ("Duke", 1, "UCLA", 8, 1), ("UConn", 1, "Iowa St", 8, 1),
        ("Houston", 1, "Purdue", 8, 1), ("Auburn", 1, "Kansas", 8, 1),
    ]
    for i, (t1, s1, t2, s2, w) in enumerate(r32_games):
        rows.append({
            "YEAR": year, "Team1": t1, "Seed1": s1,
            "Team2": t2, "Seed2": s2, "CURRENT ROUND": 32,
            "Score1": 75.0, "Score2": 70.0, "Team1_Win": w,
        })

    # S16
    rows.append({
        "YEAR": year, "Team1": "Duke", "Seed1": 1,
        "Team2": "UConn", "Seed2": 1, "CURRENT ROUND": 16,
        "Score1": 80.0, "Score2": 75.0, "Team1_Win": 1,
    })
    rows.append({
        "YEAR": year, "Team1": "Houston", "Seed1": 1,
        "Team2": "Auburn", "Seed2": 1, "CURRENT ROUND": 16,
        "Score1": 70.0, "Score2": 65.0, "Team1_Win": 1,
    })

    # E8
    rows.append({
        "YEAR": year, "Team1": "Duke", "Seed1": 1,
        "Team2": "Houston", "Seed2": 1, "CURRENT ROUND": 8,
        "Score1": 85.0, "Score2": 80.0, "Team1_Win": 1,
    })

    # F4
    rows.append({
        "YEAR": year, "Team1": "Duke", "Seed1": 1,
        "Team2": "Houston", "Seed2": 1, "CURRENT ROUND": 4,
        "Score1": 82.0, "Score2": 78.0, "Team1_Win": 1,
    })

    # NCG
    rows.append({
        "YEAR": year, "Team1": "Duke", "Seed1": 1,
        "Team2": "Houston", "Seed2": 1, "CURRENT ROUND": 2,
        "Score1": 76.0, "Score2": 72.0, "Team1_Win": 1,
    })

    return pd.DataFrame(rows)


class TestBuildActualBracket:
    def test_builds_correct_game_count(self):
        df = _make_matchups_df()
        bracket = build_actual_bracket(df, year=2024)
        assert len(bracket.games) == 17

    def test_source_is_actual(self):
        df = _make_matchups_df()
        bracket = build_actual_bracket(df, year=2024)
        assert bracket.source == "actual"

    def test_round_names_assigned(self):
        df = _make_matchups_df()
        bracket = build_actual_bracket(df, year=2024)
        round_names = {g.round_name for g in bracket.games}
        assert "R64" in round_names
        assert "NCG" in round_names

    def test_regions_assigned_for_r64(self):
        df = _make_matchups_df()
        bracket = build_actual_bracket(df, year=2024)
        r64_games = [g for g in bracket.games if g.round_name == "R64"]
        regions = {g.region for g in r64_games}
        # 8 R64 games: game_index 0-7, regions = index // 8 = all region 0
        # (A full 32-game R64 would have 4 regions)
        assert len(regions) == 1
        assert 0 in regions

    def test_winner_determined_from_team1_win(self):
        df = _make_matchups_df()
        bracket = build_actual_bracket(df, year=2024)
        # All games in our fixture have Team1_Win=1
        for game in bracket.games:
            assert game.winner == game.team1

    def test_upset_detection(self):
        """Insert an upset and verify it's detected."""
        df = _make_matchups_df()
        # Make the first game an upset: 16-seed beats 1-seed
        df.loc[0, "Team1_Win"] = 0
        bracket = build_actual_bracket(df, year=2024)
        r64_games = [g for g in bracket.games if g.round_name == "R64"]
        duke_game = [g for g in r64_games if g.team1.team == "Duke"][0]
        assert duke_game.is_upset is True
        assert duke_game.winner.team == "FDU"

    def test_filters_by_year(self):
        df2024 = _make_matchups_df(2024)
        df2023 = _make_matchups_df(2023)
        combined = pd.concat([df2024, df2023], ignore_index=True)
        bracket = build_actual_bracket(combined, year=2024)
        assert bracket.year == 2024
        # Should only have 2024 games
        assert len(bracket.games) == 17


# ---------------------------------------------------------------------------
# build_predicted_bracket
# ---------------------------------------------------------------------------

def _make_backtest_results_df(year=2024):
    """Create a minimal backtest results DataFrame (ensemble format).

    Must match the same teams/games as _make_matchups_df so the join works.
    Note: Real backtest results do NOT have CURRENT ROUND — that's recovered
    by joining back to restructured_matchups.
    """
    matchups = _make_matchups_df(year)
    rows = []
    for _, row in matchups.iterrows():
        rows.append({
            "YEAR": year,
            "Team1": row["Team1"],
            "Seed1": row["Seed1"],
            "Team2": row["Team2"],
            "Seed2": row["Seed2"],
            "Team1_Win": row["Team1_Win"],
            "Ensemble_Pred": row["Team1_Win"],  # perfect predictions
            "Ensemble_Prob": 0.85,
            "Correct_Prediction": True,
        })
    # Real results CSVs don't include CURRENT ROUND
    return pd.DataFrame(rows)


class TestBuildPredictedBracket:
    def test_builds_from_backtest_results(self):
        matchups_df = _make_matchups_df()
        results_df = _make_backtest_results_df()
        bracket = build_predicted_bracket(
            results_df, matchups_df, year=2024, source="ensemble",
        )
        assert len(bracket.games) == 17
        assert bracket.source == "ensemble"

    def test_win_probability_from_results(self):
        matchups_df = _make_matchups_df()
        results_df = _make_backtest_results_df()
        bracket = build_predicted_bracket(
            results_df, matchups_df, year=2024, source="ensemble",
        )
        # All games in our fixture have Ensemble_Prob=0.85
        for game in bracket.games:
            assert game.win_probability == 0.85

    def test_debiased_source(self):
        matchups_df = _make_matchups_df()
        # Create debiased-format results
        results_df = _make_backtest_results_df()
        results_df = results_df.rename(columns={
            "Ensemble_Pred": "Debiased_Pred",
            "Ensemble_Prob": "Debiased_Prob",
        })
        bracket = build_predicted_bracket(
            results_df, matchups_df, year=2024, source="debiased",
        )
        assert bracket.source == "debiased"
        assert len(bracket.games) == 17

    def test_incorrect_prediction_flagged(self):
        matchups_df = _make_matchups_df()
        results_df = _make_backtest_results_df()
        # Make one prediction wrong: predict Team2 wins but Team1 actually won
        results_df.loc[0, "Ensemble_Pred"] = 0
        results_df.loc[0, "Correct_Prediction"] = False
        bracket = build_predicted_bracket(
            results_df, matchups_df, year=2024, source="ensemble",
        )
        first_game = [g for g in bracket.games if g.round_name == "R64"][0]
        assert first_game.is_correct is False


# ---------------------------------------------------------------------------
# compare_brackets
# ---------------------------------------------------------------------------

class TestCompareBrackets:
    def test_comparison_adds_correctness(self):
        matchups_df = _make_matchups_df()
        actual = build_actual_bracket(matchups_df, year=2024)
        results_df = _make_backtest_results_df()
        predicted = build_predicted_bracket(
            results_df, matchups_df, year=2024, source="ensemble",
        )
        comparison = compare_brackets(predicted=predicted, actual=actual)
        # All predictions are perfect in our fixture
        for game in comparison.games:
            assert game.is_correct is True

    def test_comparison_detects_mismatch(self):
        matchups_df = _make_matchups_df()
        actual = build_actual_bracket(matchups_df, year=2024)

        # Make one prediction wrong
        results_df = _make_backtest_results_df()
        results_df.loc[0, "Ensemble_Pred"] = 0
        predicted = build_predicted_bracket(
            results_df, matchups_df, year=2024, source="ensemble",
        )
        comparison = compare_brackets(predicted=predicted, actual=actual)
        incorrect = [g for g in comparison.games if g.is_correct is False]
        assert len(incorrect) == 1

    def test_comparison_preserves_metadata(self):
        matchups_df = _make_matchups_df()
        actual = build_actual_bracket(matchups_df, year=2024)
        results_df = _make_backtest_results_df()
        predicted = build_predicted_bracket(
            results_df, matchups_df, year=2024, source="ensemble",
        )
        comparison = compare_brackets(predicted=predicted, actual=actual)
        assert comparison.year == 2024
        assert comparison.source == "ensemble_vs_actual"
