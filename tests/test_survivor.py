"""Tests for the survivor pool optimizer.

Covers the cost-matrix builder, linear-assignment solver,
bracket-aware strategy, and MC-optimal strategy using
synthetic data (no real model files or KenPom data required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports_quant.march_madness._bracket import (
    Bracket,
    BracketGame,
    BracketSlot,
    ROUND_ORDER,
)
from sports_quant.march_madness.survivor import (
    SurvivorPick,
    _build_cost_matrix,
    _build_round_candidates,
    _extract_bracket_candidates,
    _find_actual_result,
    _INFEASIBLE,
    _solve_assignment,
    run_survivor_bracket_aware,
    run_survivor_greedy,
    run_survivor_optimal,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

def _make_game_probs(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal game_probs DataFrame for testing."""
    return pd.DataFrame(rows)


def _make_bracket_game(
    round_name: str,
    game_index: int,
    t1: str,
    s1: int,
    t2: str,
    s2: int,
    winner: str,
    prob: float = 0.6,
) -> BracketGame:
    """Create a BracketGame for testing."""
    w_slot = BracketSlot(team=winner, seed=s1 if winner == t1 else s2)
    return BracketGame(
        round_name=round_name,
        region=0,
        game_index=game_index,
        team1=BracketSlot(team=t1, seed=s1),
        team2=BracketSlot(team=t2, seed=s2),
        winner=w_slot,
        win_probability=prob,
        is_upset=False,
        is_correct=None,
    )


@pytest.fixture()
def small_game_probs():
    """Two rounds, 4 teams: enough to test assignment logic."""
    return _make_game_probs([
        {
            "YEAR": 2025, "Team1": "Alpha", "Seed1": 1,
            "Team2": "Beta", "Seed2": 16,
            "CURRENT ROUND": 64, "Debiased_Prob": 0.90,
            "Team1_Win": 1,
        },
        {
            "YEAR": 2025, "Team1": "Gamma", "Seed1": 2,
            "Team2": "Delta", "Seed2": 15,
            "CURRENT ROUND": 64, "Debiased_Prob": 0.85,
            "Team1_Win": 1,
        },
        {
            "YEAR": 2025, "Team1": "Alpha", "Seed1": 1,
            "Team2": "Gamma", "Seed2": 2,
            "CURRENT ROUND": 32, "Debiased_Prob": 0.70,
            "Team1_Win": 1,
        },
    ])


@pytest.fixture()
def small_bracket():
    """Simulated bracket: Alpha beats Beta (R64), Gamma beats Delta (R64),
    then Alpha beats Gamma (R32)."""
    games = (
        _make_bracket_game("R64", 0, "Alpha", 1, "Beta", 16, "Alpha", 0.90),
        _make_bracket_game("R64", 1, "Gamma", 2, "Delta", 15, "Gamma", 0.85),
        _make_bracket_game("R32", 0, "Alpha", 1, "Gamma", 2, "Alpha", 0.70),
    )
    return Bracket(year=2025, source="simulation", games=games)


@pytest.fixture()
def actual_bracket():
    """Actual bracket matching the simulated bracket (same winners)."""
    games = (
        _make_bracket_game("R64", 0, "Alpha", 1, "Beta", 16, "Alpha", 0.90),
        _make_bracket_game("R64", 1, "Gamma", 2, "Delta", 15, "Gamma", 0.85),
        _make_bracket_game("R32", 0, "Alpha", 1, "Gamma", 2, "Gamma", 0.70),
    )
    return Bracket(year=2025, source="actual", games=games)


# ---------------------------------------------------------------------------
# Phase 1: Cost matrix and assignment solver
# ---------------------------------------------------------------------------


class TestCostMatrix:
    def test_dimensions(self, small_game_probs):
        candidates = _build_round_candidates(small_game_probs)
        rounds = [r for r in ROUND_ORDER if r in candidates]
        cost, teams, round_list = _build_cost_matrix(candidates, rounds)

        assert cost.shape == (len(teams), len(rounds))
        assert len(rounds) == 2  # R64 and R32
        assert len(teams) == 4  # Alpha, Beta, Gamma, Delta

    def test_probability_transform(self, small_game_probs):
        candidates = _build_round_candidates(small_game_probs)
        rounds = [r for r in ROUND_ORDER if r in candidates]
        cost, teams, round_list = _build_cost_matrix(candidates, rounds)

        # Alpha in R64 has prob=0.90 → cost = -log(0.90)
        alpha_idx = teams.index("Alpha")
        r64_idx = round_list.index("R64")
        expected = -np.log(0.90)
        assert abs(cost[alpha_idx, r64_idx] - expected) < 1e-10

    def test_infeasible_entries(self, small_game_probs):
        candidates = _build_round_candidates(small_game_probs)
        rounds = [r for r in ROUND_ORDER if r in candidates]
        cost, teams, round_list = _build_cost_matrix(candidates, rounds)

        # Beta is only in R64, not R32 → R32 should be infeasible
        beta_idx = teams.index("Beta")
        r32_idx = round_list.index("R32")
        assert cost[beta_idx, r32_idx] == _INFEASIBLE

    def test_zero_probability_clamped(self):
        rows = [{
            "YEAR": 2025, "Team1": "A", "Seed1": 1,
            "Team2": "B", "Seed2": 16,
            "CURRENT ROUND": 64, "Debiased_Prob": 1.0,  # B has 0.0
            "Team1_Win": 1,
        }]
        candidates = _build_round_candidates(_make_game_probs(rows))
        cost, teams, rounds = _build_cost_matrix(candidates, ["R64"])

        # B has prob=0.0 which gets clamped → should not be -inf
        b_idx = teams.index("B")
        assert np.isfinite(cost[b_idx, 0])


class TestSolveAssignment:
    def test_simple_assignment(self, small_game_probs):
        candidates = _build_round_candidates(small_game_probs)
        rounds = [r for r in ROUND_ORDER if r in candidates]
        cost, teams, round_list = _build_cost_matrix(candidates, rounds)

        assignments = _solve_assignment(cost, teams, round_list)

        # Should assign exactly 2 teams (one per round)
        assert len(assignments) == 2
        assigned_teams = {a[0] for a in assignments}
        assigned_rounds = {a[1] for a in assignments}
        assert len(assigned_teams) == 2  # no reuse
        assert assigned_rounds == {"R64", "R32"}

    def test_infeasible_filtered(self):
        # 1 team, 2 rounds — only feasible in round 1
        cost = np.array([[0.1, _INFEASIBLE]])
        assignments = _solve_assignment(cost, ["TeamA"], ["R64", "R32"])

        assert len(assignments) == 1
        assert assignments[0][1] == "R64"

    def test_round_order_sorting(self, small_game_probs):
        candidates = _build_round_candidates(small_game_probs)
        rounds = [r for r in ROUND_ORDER if r in candidates]
        cost, teams, round_list = _build_cost_matrix(candidates, rounds)
        assignments = _solve_assignment(cost, teams, round_list)

        round_names = [a[1] for a in assignments]
        assert round_names == ["R64", "R32"]


class TestOptimalStrategy:
    def test_no_candidates_returns_exhausted(self):
        empty = _make_game_probs([])
        result = run_survivor_optimal(2025, empty)

        assert result.exhausted is True
        assert len(result.picks) == 0

    def test_produces_picks(self, small_game_probs):
        result = run_survivor_optimal(2025, small_game_probs)

        assert len(result.picks) == 2
        teams_used = {p.team for p in result.picks}
        assert len(teams_used) == 2  # no reuse

    def test_maximizes_product(self, small_game_probs):
        """Optimal should pick the assignment that maximizes ∏ win_probs."""
        result = run_survivor_optimal(2025, small_game_probs)

        # The optimal assignment should use Alpha in R32 (0.70) and
        # Gamma in R64 (0.85), product = 0.595
        # vs Alpha in R64 (0.90) and Gamma in R32 (0.30), product = 0.27
        pick_map = {p.round_name: p for p in result.picks}
        assert pick_map["R64"].team == "Gamma"
        assert pick_map["R32"].team == "Alpha"


# ---------------------------------------------------------------------------
# Phase 2: Bracket-aware strategy
# ---------------------------------------------------------------------------


class TestExtractBracketCandidates:
    def test_structure(self, small_bracket):
        candidates = _extract_bracket_candidates(small_bracket)

        assert "R64" in candidates
        assert "R32" in candidates
        assert len(candidates["R64"]) == 4  # 2 games × 2 teams each... wait, 2 games
        assert len(candidates["R32"]) == 2  # 1 game × 2 teams

    def test_win_prob_complement(self, small_bracket):
        candidates = _extract_bracket_candidates(small_bracket)

        r64_cands = candidates["R64"]
        # For game 0 (Alpha vs Beta, prob=0.90):
        alpha = next(c for c in r64_cands if c["team"] == "Alpha")
        beta = next(c for c in r64_cands if c["team"] == "Beta")
        assert abs(alpha["win_prob"] - 0.90) < 1e-10
        assert abs(beta["win_prob"] - 0.10) < 1e-10


class TestFindActualResult:
    def test_team_present(self, actual_bracket):
        result = _find_actual_result("Alpha", "R64", actual_bracket)

        assert result is not None
        assert result["team"] == "Alpha"
        assert result["opponent"] == "Beta"
        assert result["actual_winner"] == "Alpha"

    def test_team_absent(self, actual_bracket):
        # Delta lost in R64 and doesn't appear in R32
        result = _find_actual_result("Delta", "R32", actual_bracket)
        assert result is None

    def test_team2_lookup(self, actual_bracket):
        result = _find_actual_result("Beta", "R64", actual_bracket)

        assert result is not None
        assert result["opponent"] == "Alpha"
        assert result["actual_winner"] == "Alpha"


class TestBracketAware:
    def test_reserves_deep_team(self):
        """A team predicted to go deep should be saved for their deepest round.

        Setup: StarTeam is the only team that reaches S16 as the favoured
        side (0.80). There are plenty of strong options in R64 and R32,
        so the optimizer should save StarTeam for S16 where no other
        strong alternative exists.
        """
        games = (
            # R64: StarTeam and StrongR64 both win easily
            _make_bracket_game("R64", 0, "StarTeam", 1, "Fodder1", 16, "StarTeam", 0.95),
            _make_bracket_game("R64", 1, "StrongR64", 2, "Fodder2", 15, "StrongR64", 0.93),
            _make_bracket_game("R64", 2, "StrongR32", 3, "Fodder3", 14, "StrongR32", 0.88),
            # R32: StrongR32 is a viable pick (0.80 win prob)
            _make_bracket_game("R32", 0, "StarTeam", 1, "StrongR64", 2, "StarTeam", 0.70),
            _make_bracket_game("R32", 1, "StrongR32", 3, "Opponent4", 6, "StrongR32", 0.80),
            # S16: only StarTeam is strong here
            _make_bracket_game("S16", 0, "StarTeam", 1, "StrongR32", 3, "StarTeam", 0.75),
        )
        sim_bracket = Bracket(year=2025, source="simulation", games=games)
        actual_bracket = Bracket(year=2025, source="actual", games=games)

        result = run_survivor_bracket_aware(2025, sim_bracket, actual_bracket)

        pick_map = {p.round_name: p.team for p in result.picks}
        # StarTeam should be saved for S16, not wasted in R64
        assert pick_map.get("S16") == "StarTeam"
        assert pick_map.get("R64") != "StarTeam"

    def test_team_eliminated_early(self):
        """If the actual bracket doesn't match simulation, pick should fail."""
        sim_games = (
            _make_bracket_game("R64", 0, "A", 1, "B", 16, "A", 0.90),
            _make_bracket_game("R32", 0, "A", 1, "C", 8, "A", 0.70),
        )
        sim_bracket = Bracket(year=2025, source="simulation", games=sim_games)

        # In reality, A lost in R64
        actual_games = (
            _make_bracket_game("R64", 0, "A", 1, "B", 16, "B", 0.90),
        )
        actual_bracket = Bracket(year=2025, source="actual", games=actual_games)

        result = run_survivor_bracket_aware(2025, sim_bracket, actual_bracket)

        # A is assigned to R32 by optimizer but was eliminated in R64
        for pick in result.picks:
            if pick.round_name == "R32" and pick.team == "A":
                assert pick.survived is False

    def test_strategy_name(self, small_bracket, actual_bracket):
        result = run_survivor_bracket_aware(2025, small_bracket, actual_bracket)
        assert result.strategy == "bracket_aware"
