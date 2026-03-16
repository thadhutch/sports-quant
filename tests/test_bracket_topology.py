"""Tests for bracket topology constraints.

Covers team-to-side mapping, the constrained ILP solver, and
integration with the survivor optimizer to ensure picks are
distributed across both bracket halves.
"""

from __future__ import annotations

import numpy as np
import pytest

from sports_quant.march_madness._bracket import (
    Bracket,
    BracketGame,
    BracketSlot,
    ROUND_ORDER,
)
from sports_quant.march_madness._bracket_topology import (
    _max_regional_picks_per_side,
    _solve_unconstrained,
    build_team_side_map,
    build_team_side_map_from_r64,
    solve_constrained_assignment,
)
from sports_quant.march_madness.survivor import (
    _build_cost_matrix,
    _extract_bracket_candidates,
    _INFEASIBLE,
    run_survivor_bracket_aware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game(
    round_name: str,
    game_index: int,
    t1: str,
    s1: int,
    t2: str,
    s2: int,
    winner: str,
    prob: float = 0.6,
    region: int | None = None,
) -> BracketGame:
    """Create a BracketGame with explicit or auto-computed region."""
    from sports_quant.march_madness._bracket_builder import _assign_region

    if region is None:
        region = _assign_region(round_name, game_index)
    w_slot = BracketSlot(team=winner, seed=s1 if winner == t1 else s2)
    return BracketGame(
        round_name=round_name,
        region=region,
        game_index=game_index,
        team1=BracketSlot(team=t1, seed=s1),
        team2=BracketSlot(team=t2, seed=s2),
        winner=w_slot,
        win_probability=prob,
        is_upset=False,
        is_correct=None,
    )


class FakeTeamStats:
    """Minimal stand-in for TeamStats in R64 matchup lists."""

    def __init__(self, team: str, seed: int = 1):
        self.team = team
        self.seed = seed


# ---------------------------------------------------------------------------
# Team-to-side mapping
# ---------------------------------------------------------------------------


class TestBuildTeamSideMap:
    def test_from_bracket_with_regions(self):
        """R64 games with correct regions map teams to sides."""
        games = (
            _make_game("R64", 0, "A", 1, "B", 16, "A", region=0),
            _make_game("R64", 8, "C", 1, "D", 16, "C", region=1),
            _make_game("R64", 16, "E", 1, "F", 16, "E", region=2),
            _make_game("R64", 24, "G", 1, "H", 16, "G", region=3),
        )
        bracket = Bracket(year=2025, source="test", games=games)
        side_map = build_team_side_map(bracket)

        assert side_map["A"] == "left"
        assert side_map["B"] == "left"
        assert side_map["C"] == "left"
        assert side_map["D"] == "left"
        assert side_map["E"] == "right"
        assert side_map["F"] == "right"
        assert side_map["G"] == "right"
        assert side_map["H"] == "right"

    def test_ignores_non_r64_games(self):
        games = (
            _make_game("R64", 0, "A", 1, "B", 16, "A", region=0),
            _make_game("R32", 0, "A", 1, "C", 8, "A", region=0),
        )
        bracket = Bracket(year=2025, source="test", games=games)
        side_map = build_team_side_map(bracket)

        assert "A" in side_map
        assert "B" in side_map
        assert "C" not in side_map  # only appears in R32


class TestBuildTeamSideMapFromR64:
    def test_32_matchups(self):
        """32 R64 matchups: games 0-15 = left, 16-31 = right."""
        matchups = []
        for i in range(32):
            t1 = FakeTeamStats(f"Team{i*2}", seed=1)
            t2 = FakeTeamStats(f"Team{i*2+1}", seed=16)
            matchups.append((t1, t2))

        side_map = build_team_side_map_from_r64(matchups)

        # Games 0-15 (regions 0,1) → left
        for i in range(16):
            assert side_map[f"Team{i*2}"] == "left"
            assert side_map[f"Team{i*2+1}"] == "left"

        # Games 16-31 (regions 2,3) → right
        for i in range(16, 32):
            assert side_map[f"Team{i*2}"] == "right"
            assert side_map[f"Team{i*2+1}"] == "right"


# ---------------------------------------------------------------------------
# Budget calculation
# ---------------------------------------------------------------------------


class TestMaxRegionalPicksPerSide:
    def test_full_bracket(self):
        # 4 regional rounds → at most 3 from one side
        assert _max_regional_picks_per_side(4) == 3

    def test_two_rounds(self):
        assert _max_regional_picks_per_side(2) == 1

    def test_one_round(self):
        assert _max_regional_picks_per_side(1) == 1


# ---------------------------------------------------------------------------
# Constrained ILP solver
# ---------------------------------------------------------------------------


def _build_imbalanced_scenario():
    """Build a 6-round scenario where left side dominates.

    Left-side teams (L1-L4) have very high win probs in all rounds.
    Right-side teams (R1-R4) have moderate win probs.

    Without constraints, the optimizer would pick 4 left-side teams
    in R64-E8, leaving no unused left-side team for F4.
    """
    teams = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]
    rounds = list(ROUND_ORDER)
    side_map = {
        "L1": "left", "L2": "left", "L3": "left", "L4": "left",
        "R1": "right", "R2": "right", "R3": "right", "R4": "right",
    }

    # Cost matrix: -log(win_prob)
    # Left-side teams: low cost (high prob) across all rounds
    # Right-side teams: higher cost (lower prob)
    cost = np.full((8, 6), _INFEASIBLE)

    # L1: strong in R64 (0.97), feasible in F4 (0.65)
    cost[0, 0] = -np.log(0.97)  # R64
    cost[0, 4] = -np.log(0.65)  # F4

    # L2: strong in R32 (0.95)
    cost[1, 1] = -np.log(0.95)  # R32

    # L3: strong in S16 (0.92)
    cost[2, 2] = -np.log(0.92)  # S16

    # L4: strong in E8 (0.90)
    cost[3, 3] = -np.log(0.90)  # E8

    # R1: decent in R64 (0.80), feasible in F4 (0.60)
    cost[4, 0] = -np.log(0.80)  # R64
    cost[4, 4] = -np.log(0.60)  # F4

    # R2: decent in R32 (0.78), feasible in NCG (0.55)
    cost[5, 1] = -np.log(0.78)  # R32
    cost[5, 5] = -np.log(0.55)  # NCG

    # R3: decent in S16 (0.75), feasible in NCG (0.50)
    cost[6, 2] = -np.log(0.75)  # S16
    cost[6, 5] = -np.log(0.50)  # NCG

    # R4: decent in E8 (0.72)
    cost[7, 3] = -np.log(0.72)  # E8

    return cost, teams, rounds, side_map


class TestConstrainedAssignment:
    def test_side_balance_enforced(self):
        """Constrained solver limits one side to at most 3 picks in R64-E8."""
        cost, teams, rounds, side_map = _build_imbalanced_scenario()

        assignments = solve_constrained_assignment(
            cost, teams, rounds, side_map,
        )

        # Count left-side picks in regional rounds
        regional = {"R64", "R32", "S16", "E8"}
        left_regional = sum(
            1 for t, r, _ in assignments
            if r in regional and side_map.get(t) == "left"
        )
        right_regional = sum(
            1 for t, r, _ in assignments
            if r in regional and side_map.get(t) == "right"
        )

        assert left_regional <= 3, f"Left has {left_regional} regional picks"
        assert right_regional <= 3, f"Right has {right_regional} regional picks"

    def test_all_rounds_covered(self):
        """Constrained solver should still assign all 6 rounds."""
        cost, teams, rounds, side_map = _build_imbalanced_scenario()

        assignments = solve_constrained_assignment(
            cost, teams, rounds, side_map,
        )
        assigned_rounds = {r for _, r, _ in assignments}

        assert assigned_rounds == set(ROUND_ORDER)

    def test_unconstrained_would_imbalance(self):
        """Verify the test scenario: unconstrained over-concentrates on left."""
        cost, teams, rounds, side_map = _build_imbalanced_scenario()

        unconstrained = _solve_unconstrained(cost, teams, rounds)

        regional = {"R64", "R32", "S16", "E8"}
        left_regional = sum(
            1 for t, r, _ in unconstrained
            if r in regional and side_map.get(t) == "left"
        )

        # Unconstrained should pick all 4 left in R64-E8 (they're cheapest)
        assert left_regional == 4, (
            f"Expected unconstrained to pick 4 left, got {left_regional}"
        )

    def test_no_team_reuse(self):
        """Each team should appear at most once in assignments."""
        cost, teams, rounds, side_map = _build_imbalanced_scenario()

        assignments = solve_constrained_assignment(
            cost, teams, rounds, side_map,
        )
        assigned_teams = [t for t, _, _ in assignments]

        assert len(assigned_teams) == len(set(assigned_teams))


class TestConstrainedFallback:
    def test_fallback_on_empty_side(self):
        """When all teams are on one side, falls back to unconstrained."""
        teams = ["A", "B"]
        rounds = ["R64", "R32"]
        side_map = {"A": "left", "B": "left"}  # no right-side teams
        cost = np.array([
            [-np.log(0.9), _INFEASIBLE],
            [_INFEASIBLE, -np.log(0.8)],
        ])

        # Should not crash — falls back gracefully
        assignments = solve_constrained_assignment(
            cost, teams, rounds, side_map,
        )
        assert len(assignments) == 2

    def test_single_round_skips_constraints(self):
        """With only 1 round, side constraints don't apply."""
        teams = ["A", "B"]
        rounds = ["NCG"]
        side_map = {"A": "left", "B": "right"}
        cost = np.array([
            [-np.log(0.7)],
            [-np.log(0.6)],
        ])

        assignments = solve_constrained_assignment(
            cost, teams, rounds, side_map,
        )
        assert len(assignments) == 1
        assert assignments[0][0] == "A"  # higher prob


class TestConstrainedWithPriorCounts:
    def test_prior_picks_reduce_budget(self):
        """Prior left-side picks reduce remaining left budget."""
        cost, teams, rounds, side_map = _build_imbalanced_scenario()

        # Pretend 2 left-side picks already made in regional rounds
        prior = {"left": 2, "right": 0}

        assignments = solve_constrained_assignment(
            cost, teams, rounds, side_map,
            prior_side_counts=prior,
        )

        regional = {"R64", "R32", "S16", "E8"}
        left_regional = sum(
            1 for t, r, _ in assignments
            if r in regional and side_map.get(t) == "left"
        )

        # Budget was 3, minus 2 prior = at most 1 more left in regional
        assert left_regional <= 1


class TestBalancedScenarioUnchanged:
    def test_already_balanced_same_result(self):
        """When unconstrained is balanced, constrained gives same cost."""
        teams = ["L1", "L2", "R1", "R2"]
        rounds = ["R64", "R32", "S16", "E8"]
        side_map = {
            "L1": "left", "L2": "left",
            "R1": "right", "R2": "right",
        }

        # Each team is best in exactly one round, naturally balanced
        cost = np.full((4, 4), _INFEASIBLE)
        cost[0, 0] = -np.log(0.95)  # L1 in R64
        cost[1, 1] = -np.log(0.90)  # L2 in R32
        cost[2, 2] = -np.log(0.85)  # R1 in S16
        cost[3, 3] = -np.log(0.80)  # R2 in E8

        constrained = solve_constrained_assignment(
            cost, teams, rounds, side_map,
        )
        unconstrained = _solve_unconstrained(cost, teams, rounds)

        # Same teams in same rounds
        c_set = {(t, r) for t, r, _ in constrained}
        u_set = {(t, r) for t, r, _ in unconstrained}
        assert c_set == u_set


# ---------------------------------------------------------------------------
# Integration: bracket-aware strategy with topology constraints
# ---------------------------------------------------------------------------


class TestBracketAwareConstrained:
    def _build_full_bracket(self, all_left_dominant: bool = True):
        """Build a 6-round bracket with teams on both sides.

        When all_left_dominant=True, left-side teams are much stronger,
        testing whether the constraint prevents left-side over-concentration.
        """
        games = []
        game_idx = 0

        # R64: 32 games (8 per region)
        for region in range(4):
            side = "left" if region < 2 else "right"
            for i in range(8):
                t1 = f"{side[0].upper()}{region}_{i}_A"
                t2 = f"{side[0].upper()}{region}_{i}_B"
                s1, s2 = i + 1, 16 - i
                prob = 0.95 if side == "left" else 0.70
                games.append(_make_game(
                    "R64", game_idx, t1, s1, t2, s2, t1, prob,
                ))
                game_idx += 1

        # R32: 16 games (4 per region)
        game_idx = 0
        for region in range(4):
            side = "left" if region < 2 else "right"
            for i in range(4):
                t1 = f"{side[0].upper()}{region}_{i*2}_A"
                t2 = f"{side[0].upper()}{region}_{i*2+1}_A"
                prob = 0.90 if side == "left" else 0.65
                games.append(_make_game(
                    "R32", game_idx, t1, 1, t2, 2, t1, prob,
                ))
                game_idx += 1

        # S16: 8 games (2 per region)
        game_idx = 0
        for region in range(4):
            side = "left" if region < 2 else "right"
            for i in range(2):
                t1 = f"{side[0].upper()}{region}_{i*4}_A"
                t2 = f"{side[0].upper()}{region}_{i*4+2}_A"
                prob = 0.85 if side == "left" else 0.60
                games.append(_make_game(
                    "S16", game_idx, t1, 1, t2, 2, t1, prob,
                ))
                game_idx += 1

        # E8: 4 games (1 per region)
        game_idx = 0
        for region in range(4):
            side = "left" if region < 2 else "right"
            t1 = f"{side[0].upper()}{region}_0_A"
            t2 = f"{side[0].upper()}{region}_4_A"
            prob = 0.80 if side == "left" else 0.55
            games.append(_make_game(
                "E8", game_idx, t1, 1, t2, 2, t1, prob,
            ))
            game_idx += 1

        # F4: 2 games (cross-region)
        games.append(_make_game(
            "F4", 0, "L0_0_A", 1, "L1_0_A", 1, "L0_0_A", 0.75,
            region=-1,
        ))
        games.append(_make_game(
            "F4", 1, "R2_0_A", 1, "R3_0_A", 1, "R2_0_A", 0.55,
            region=-1,
        ))

        # NCG
        games.append(_make_game(
            "NCG", 0, "L0_0_A", 1, "R2_0_A", 1, "L0_0_A", 0.65,
            region=-1,
        ))

        return Bracket(year=2025, source="simulation", games=tuple(games))

    def test_constrained_preserves_both_sides(self):
        """With left-dominant bracket, constrained solver keeps right options."""
        bracket = self._build_full_bracket(all_left_dominant=True)
        actual = bracket  # same winners for simplicity

        result = run_survivor_bracket_aware(2025, bracket, actual)

        # Check side distribution in regional rounds
        side_map = build_team_side_map(bracket)
        regional = {"R64", "R32", "S16", "E8"}
        left_count = sum(
            1 for p in result.picks
            if p.round_name in regional and side_map.get(p.team) == "left"
        )
        right_count = sum(
            1 for p in result.picks
            if p.round_name in regional and side_map.get(p.team) == "right"
        )

        assert left_count <= 3, f"Left regional picks: {left_count}"
        assert right_count <= 3, f"Right regional picks: {right_count}"
        assert left_count + right_count == 4


class TestILPPerformance:
    def test_solve_time_reasonable(self):
        """64x6 ILP should solve in under 1 second."""
        import time

        n_teams = 64
        n_rounds = 6
        teams = [f"Team{i}" for i in range(n_teams)]
        rounds = list(ROUND_ORDER)
        side_map = {
            f"Team{i}": "left" if i < 32 else "right"
            for i in range(n_teams)
        }

        rng = np.random.default_rng(42)
        cost = np.full((n_teams, n_rounds), _INFEASIBLE)
        # Make ~10 teams feasible per round
        for j in range(n_rounds):
            feasible = rng.choice(n_teams, size=10, replace=False)
            for i in feasible:
                cost[i, j] = -np.log(rng.uniform(0.5, 0.95))

        start = time.monotonic()
        solve_constrained_assignment(cost, teams, rounds, side_map)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"ILP took {elapsed:.2f}s"
