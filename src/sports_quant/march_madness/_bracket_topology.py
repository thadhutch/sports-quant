"""Bracket topology constraints for survivor pool optimization.

Maps teams to bracket sides (left/right) and provides a constrained
ILP solver that ensures picks are distributed across both halves of
the bracket, preserving viable options for F4 and NCG rounds.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

from sports_quant.march_madness._bracket import (
    Bracket,
    REGIONAL_ROUNDS,
    REGIONAL_SURVIVOR_SLOTS,
    ROUND_ORDER,
    SIDE_FOR_REGION,
    SLOT_TO_ROUND,
    SURVIVOR_SLOTS,
)

logger = logging.getLogger(__name__)

_INFEASIBLE = 1e9
_PROB_FLOOR = 1e-10

# Pre-computed ordering for sorting assignments (supports both
# round names like "R64" and survivor slots like "R64_D1")
_ROUND_ORDER_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(ROUND_ORDER)
}
_ROUND_ORDER_INDEX.update({
    name: idx for idx, name in enumerate(SURVIVOR_SLOTS)
})

# Set of all regional column names (round names + survivor slot names)
_REGIONAL_COLUMNS: frozenset[str] = frozenset(REGIONAL_ROUNDS) | frozenset(
    REGIONAL_SURVIVOR_SLOTS
)


# ---------------------------------------------------------------------------
# Team-to-side mapping
# ---------------------------------------------------------------------------


def build_team_side_map(bracket: Bracket) -> dict[str, str]:
    """Map each team to its bracket side based on R64 region.

    Args:
        bracket: A Bracket object with games that have region info.

    Returns:
        Dict of team_name -> "left" or "right".
    """
    side_map: dict[str, str] = {}
    for game in bracket.games:
        if game.round_name != "R64":
            continue
        side = SIDE_FOR_REGION.get(game.region)
        if side is None:
            continue
        side_map[game.team1.team] = side
        side_map[game.team2.team] = side
    return side_map


def build_team_side_map_from_r64(
    r64_matchups: list[tuple],
) -> dict[str, str]:
    """Map each team to its bracket side from R64 matchup list.

    R64 matchups are in canonical bracket order: games 0-7 = region 0,
    8-15 = region 1 (both "left"), 16-23 = region 2, 24-31 = region 3
    (both "right").

    Args:
        r64_matchups: List of 32 (TeamStats, TeamStats) pairs in
            canonical bracket order.

    Returns:
        Dict of team_name -> "left" or "right".
    """
    side_map: dict[str, str] = {}
    for game_idx, (t1, t2) in enumerate(r64_matchups):
        region = game_idx // 8
        side = SIDE_FOR_REGION[region]
        side_map[t1.team] = side
        side_map[t2.team] = side
    return side_map


# ---------------------------------------------------------------------------
# Constrained ILP solver
# ---------------------------------------------------------------------------


def _max_regional_picks_per_side(
    n_regional_rounds: int,
) -> int:
    """Maximum picks allowed from one bracket side in regional rounds.

    With ``n`` regional rounds (R64-E8), we allow at most ``n - 1``
    picks from one side, ensuring at least 1 pick from each side is
    reserved for F4+NCG.
    """
    return max(n_regional_rounds - 1, 1)


def solve_constrained_assignment(
    cost_matrix: np.ndarray,
    teams: list[str],
    rounds: list[str],
    team_side_map: dict[str, str],
    prior_side_counts: dict[str, int] | None = None,
) -> list[tuple[str, str, float]]:
    """Solve the survivor assignment with bracket-side constraints.

    Uses an ILP to minimize total ``-log(win_prob)`` cost while
    ensuring picks don't over-concentrate on one bracket side in
    R64-E8, preserving viable options for F4 and NCG.

    Constraints:
        - Each team used at most once
        - Each round gets exactly one pick
        - Infeasible cells (team not playing) are excluded
        - At most ``max_per_side`` picks from either bracket side
          in regional rounds (R64-E8), accounting for prior picks

    Falls back to unconstrained ``linear_sum_assignment`` if the
    ILP is infeasible.

    Args:
        cost_matrix: (n_teams, n_rounds) cost matrix with -log(p) values.
            Infeasible entries have value >= ``_INFEASIBLE - 1``.
        teams: Team names corresponding to rows.
        rounds: Round names corresponding to columns.
        team_side_map: Dict of team_name -> "left" or "right".
        prior_side_counts: When solving sequentially, the number of
            picks already committed from each side in regional rounds.
            E.g. ``{"left": 1, "right": 2}`` means 1 left-side and
            2 right-side picks already made in R64-E8.

    Returns:
        List of (team, round_name, win_prob) sorted by round order,
        excluding infeasible assignments.
    """
    n_teams, n_rounds = cost_matrix.shape

    if n_teams == 0 or n_rounds == 0:
        return []

    # Identify which columns are regional (R64-E8 rounds or day slots)
    regional_col_indices = [
        j for j, r in enumerate(rounds) if r in _REGIONAL_COLUMNS
    ]
    n_regional = len(regional_col_indices)

    # If no regional rounds remain, or only 1 round total, skip constraints
    if n_regional <= 1 or n_rounds <= 1:
        return _solve_unconstrained(cost_matrix, teams, rounds)

    # Build side membership vectors for teams
    left_teams = {
        i for i, t in enumerate(teams)
        if team_side_map.get(t) == "left"
    }
    right_teams = {
        i for i, t in enumerate(teams)
        if team_side_map.get(t) == "right"
    }

    # If we can't determine sides for enough teams, fall back
    if not left_teams or not right_teams:
        logger.warning(
            "Cannot determine bracket sides for teams — "
            "falling back to unconstrained solver",
        )
        return _solve_unconstrained(cost_matrix, teams, rounds)

    # Compute budget: max picks per side in remaining regional slots.
    # Detect whether we're using survivor day-slots (7 regional) or
    # plain round names (4 regional). Day-slots contain "_D" suffix
    # (e.g. R64_D1, R32_D2) — their presence signals 9-slot mode.
    prior_left = (prior_side_counts or {}).get("left", 0)
    prior_right = (prior_side_counts or {}).get("right", 0)
    uses_survivor_slots = any("_D" in r for r in rounds)
    total_regional = (
        len(REGIONAL_SURVIVOR_SLOTS) if uses_survivor_slots
        else len(REGIONAL_ROUNDS)
    )
    max_per_side = _max_regional_picks_per_side(total_regional)
    budget_left = max(max_per_side - prior_left, 0)
    budget_right = max(max_per_side - prior_right, 0)

    # If budgets are already exhausted or trivially non-binding, skip
    if budget_left >= n_regional and budget_right >= n_regional:
        return _solve_unconstrained(cost_matrix, teams, rounds)

    result = _solve_ilp(
        cost_matrix, teams, rounds,
        left_teams, right_teams,
        regional_col_indices,
        budget_left, budget_right,
    )

    if result is not None:
        return result

    logger.warning(
        "Constrained ILP infeasible — falling back to unconstrained solver",
    )
    return _solve_unconstrained(cost_matrix, teams, rounds)


def _solve_unconstrained(
    cost_matrix: np.ndarray,
    teams: list[str],
    rounds: list[str],
) -> list[tuple[str, str, float]]:
    """Solve with standard linear_sum_assignment (no side constraints)."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments: list[tuple[str, str, float]] = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= _INFEASIBLE - 1.0:
            continue
        win_prob = float(np.exp(-cost_matrix[r, c]))
        assignments.append((teams[r], rounds[c], win_prob))

    assignments.sort(key=lambda a: _ROUND_ORDER_INDEX.get(a[1], 99))
    return assignments


def _solve_ilp(
    cost_matrix: np.ndarray,
    teams: list[str],
    rounds: list[str],
    left_teams: set[int],
    right_teams: set[int],
    regional_col_indices: list[int],
    budget_left: int,
    budget_right: int,
) -> list[tuple[str, str, float]] | None:
    """Solve the bracket-constrained assignment as an ILP.

    Decision variables: x[i * n_rounds + j] = 1 if team i is
    assigned to round j.

    Returns:
        Sorted list of (team, round, win_prob), or None if infeasible.
    """
    n_teams, n_rounds = cost_matrix.shape
    n_vars = n_teams * n_rounds

    # Flatten cost matrix to 1D objective vector
    c = cost_matrix.flatten().astype(np.float64)

    # All variables are binary
    integrality = np.ones(n_vars, dtype=int)

    # Variable bounds: fix infeasible cells to 0
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for i in range(n_teams):
        for j in range(n_rounds):
            if cost_matrix[i, j] >= _INFEASIBLE - 1.0:
                ub[i * n_rounds + j] = 0.0

    constraints: list[LinearConstraint] = []

    # Constraint 1: each round gets exactly one pick
    # sum_i x[i, j] == 1 for each j
    for j in range(n_rounds):
        a = np.zeros(n_vars)
        for i in range(n_teams):
            a[i * n_rounds + j] = 1.0
        constraints.append(LinearConstraint(a, lb=1.0, ub=1.0))

    # Constraint 2: each team used at most once
    # sum_j x[i, j] <= 1 for each i
    for i in range(n_teams):
        a = np.zeros(n_vars)
        for j in range(n_rounds):
            a[i * n_rounds + j] = 1.0
        constraints.append(LinearConstraint(a, lb=0.0, ub=1.0))

    # Constraint 3: left-side budget in regional rounds
    # sum_{i in left, j in regional} x[i,j] <= budget_left
    a_left = np.zeros(n_vars)
    for i in left_teams:
        for j in regional_col_indices:
            a_left[i * n_rounds + j] = 1.0
    constraints.append(LinearConstraint(a_left, lb=0.0, ub=float(budget_left)))

    # Constraint 4: right-side budget in regional rounds
    a_right = np.zeros(n_vars)
    for i in right_teams:
        for j in regional_col_indices:
            a_right[i * n_rounds + j] = 1.0
    constraints.append(
        LinearConstraint(a_right, lb=0.0, ub=float(budget_right)),
    )

    solution = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb=lb, ub=ub),
    )

    if not solution.success:
        return None

    # Extract assignments from solution
    x_vals = solution.x
    assignments: list[tuple[str, str, float]] = []
    for i in range(n_teams):
        for j in range(n_rounds):
            if x_vals[i * n_rounds + j] > 0.5:  # binary threshold
                if cost_matrix[i, j] < _INFEASIBLE - 1.0:
                    win_prob = float(np.exp(-cost_matrix[i, j]))
                    assignments.append((teams[i], rounds[j], win_prob))

    assignments.sort(key=lambda a: _ROUND_ORDER_INDEX.get(a[1], 99))

    # Log the side distribution
    team_idx_lookup = {t: i for i, t in enumerate(teams)}
    side_dist: dict[str, int] = {"left": 0, "right": 0}
    for team_name, round_name, _ in assignments:
        if round_name in _REGIONAL_COLUMNS:
            idx = team_idx_lookup.get(team_name)
            if idx is not None:
                if idx in left_teams:
                    side_dist["left"] += 1
                elif idx in right_teams:
                    side_dist["right"] += 1

    logger.info(
        "Constrained assignment: left=%d, right=%d in regional rounds",
        side_dist["left"], side_dist["right"],
    )

    return assignments
