"""Survivor pool optimizer for March Madness.

Picks one team per round to win their game, with no team reuse.
Supports greedy, optimal (linear assignment), bracket-aware, and
Monte Carlo optimal strategies. Also provides a live prediction mode
that combines known results with forward simulation of future rounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sports_quant.march_madness._bracket import (
    Bracket,
    REGIONAL_ROUNDS,
    REGIONAL_SURVIVOR_SLOTS,
    ROUND_NAMES,
    ROUND_ORDER,
    SLOT_TO_ROUND,
    SURVIVOR_SLOTS,
)
from sports_quant.march_madness._bracket_topology import (
    build_team_side_map,
    build_team_side_map_from_r64,
    solve_constrained_assignment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivorPick:
    """A single survivor pool pick."""

    round_name: str
    team: str
    seed: int
    opponent: str
    opponent_seed: int
    win_probability: float
    actual_winner: str
    survived: bool
    day_slot: str | None = None  # Survivor slot: R64_D1, R64_D2, etc.


@dataclass(frozen=True)
class SurvivorResult:
    """Complete survivor pool result for one tournament year."""

    year: int
    strategy: str
    picks: tuple[SurvivorPick, ...]
    survived_all: bool
    rounds_survived: int
    total_rounds: int
    survival_probability: float
    exhausted: bool  # True when no candidate was available for a round


@dataclass(frozen=True)
class SurvivorMonteCarloResult:
    """Aggregated survivor results from N Monte Carlo simulations."""

    year: int
    strategy: str
    n_simulations: int
    survival_rate: float
    mean_rounds_survived: float
    rounds_survived_distribution: dict[int, float]
    team_pick_rates: dict[str, dict[str, float]]
    elimination_round_distribution: dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_round_candidates(
    game_probs: pd.DataFrame,
    use_day_slots: bool = False,
) -> dict[str, list[dict]]:
    """Build per-slot candidate lists from game probabilities.

    Each candidate is a dict with team info and win probability for
    both the Team1 side and Team2 side of each game.

    Args:
        game_probs: DataFrame with YEAR, Team1, Seed1, Team2, Seed2,
                    CURRENT ROUND, Debiased_Prob, Team1_Win.
                    When ``use_day_slots=True``, must also have ``day_slot``.
        use_day_slots: If True, key candidates by ``day_slot`` column
                       (e.g. R64_D1) instead of round name (e.g. R64).

    Returns:
        slot_or_round -> list of candidate dicts, each with keys:
        team, seed, opponent, opponent_seed, win_prob, actual_winner,
        round_name, day_slot.
    """
    candidates: dict[str, list[dict]] = {}

    for _, row in game_probs.iterrows():
        round_num = int(row["CURRENT ROUND"])
        round_name = ROUND_NAMES.get(round_num)
        if round_name is None:
            continue

        # Determine the grouping key
        if use_day_slots and "day_slot" in row.index and pd.notna(row.get("day_slot")):
            key = str(row["day_slot"])
        else:
            key = round_name

        prob = float(row["Debiased_Prob"])
        actual_winner = (
            row["Team1"] if int(row["Team1_Win"]) == 1 else row["Team2"]
        )

        day_slot = str(row["day_slot"]) if "day_slot" in row.index and pd.notna(row.get("day_slot")) else None

        candidates.setdefault(key, [])

        base = {
            "round_name": round_name,
            "day_slot": day_slot,
        }

        # Team1 candidate
        candidates[key].append({
            **base,
            "team": row["Team1"],
            "seed": int(row["Seed1"]),
            "opponent": row["Team2"],
            "opponent_seed": int(row["Seed2"]),
            "win_prob": prob,
            "actual_winner": actual_winner,
        })

        # Team2 candidate
        candidates[key].append({
            **base,
            "team": row["Team2"],
            "seed": int(row["Seed2"]),
            "opponent": row["Team1"],
            "opponent_seed": int(row["Seed1"]),
            "win_prob": 1.0 - prob,
            "actual_winner": actual_winner,
        })

    return candidates


_INFEASIBLE = 1e9
_PROB_FLOOR = 1e-10


def _build_cost_matrix(
    candidates: dict[str, list[dict]],
    rounds: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a cost matrix for the linear assignment solver.

    Rows are teams, columns are rounds. Cost is ``-log(win_prob)``
    so that minimising the sum of costs maximises the product of
    win probabilities. Infeasible cells (team not playing in that
    round) get a large sentinel value.

    Returns:
        (cost_matrix, team_list, round_list)
    """
    all_teams: set[str] = set()
    for round_name in rounds:
        for c in candidates.get(round_name, []):
            all_teams.add(c["team"])
    teams = sorted(all_teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    cost = np.full((len(teams), len(rounds)), _INFEASIBLE)

    for j, round_name in enumerate(rounds):
        for c in candidates.get(round_name, []):
            i = team_idx[c["team"]]
            p = max(c["win_prob"], _PROB_FLOOR)
            cell_cost = -np.log(p)
            # Keep the best (lowest cost) if a team appears multiple times
            if cell_cost < cost[i, j]:
                cost[i, j] = cell_cost

    return cost, teams, rounds


def _solve_assignment(
    cost_matrix: np.ndarray,
    teams: list[str],
    rounds: list[str],
) -> list[tuple[str, str, float]]:
    """Solve the linear assignment problem on the cost matrix.

    Returns:
        List of (team, round_name, win_prob) sorted by round order,
        excluding infeasible assignments.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments: list[tuple[str, str, float]] = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= _INFEASIBLE - 1.0:
            continue
        win_prob = float(np.exp(-cost_matrix[r, c]))
        assignments.append((teams[r], rounds[c], win_prob))

    round_order = {name: idx for idx, name in enumerate(ROUND_ORDER)}
    assignments.sort(key=lambda x: round_order.get(x[1], 99))
    return assignments


def _find_actual_result(
    team: str,
    round_name: str,
    actual_bracket: Bracket,
) -> dict | None:
    """Look up a team's actual game result in a given round.

    Returns:
        Candidate-style dict with opponent info and actual_winner,
        or None if the team did not play in that round.
    """
    for game in actual_bracket.games:
        if game.round_name != round_name:
            continue
        if game.winner is None:
            continue
        if game.team1.team == team:
            return {
                "team": team,
                "seed": game.team1.seed,
                "opponent": game.team2.team,
                "opponent_seed": game.team2.seed,
                "win_prob": game.win_probability or 0.5,
                "actual_winner": game.winner.team,
            }
        if game.team2.team == team:
            return {
                "team": team,
                "seed": game.team2.seed,
                "opponent": game.team1.team,
                "opponent_seed": game.team1.seed,
                "win_prob": (
                    1.0 - game.win_probability
                    if game.win_probability is not None
                    else 0.5
                ),
                "actual_winner": game.winner.team,
            }
    return None


def _make_pick(
    candidate: dict,
    round_name: str,
    day_slot: str | None = None,
) -> SurvivorPick:
    """Create a SurvivorPick from a candidate dict."""
    survived = candidate["actual_winner"] == candidate["team"]
    return SurvivorPick(
        round_name=round_name,
        team=candidate["team"],
        seed=candidate["seed"],
        opponent=candidate["opponent"],
        opponent_seed=candidate["opponent_seed"],
        win_probability=candidate["win_prob"],
        actual_winner=candidate["actual_winner"],
        survived=survived,
        day_slot=day_slot or candidate.get("day_slot"),
    )


def _result_from_picks(
    year: int,
    strategy: str,
    picks: list[SurvivorPick],
    total_rounds: int | None = None,
    exhausted: bool = False,
) -> SurvivorResult:
    """Build a SurvivorResult from a list of picks."""
    # Infer total from picks: if any pick has day_slot, use 9 slots
    if total_rounds is None:
        has_day_slots = any(p.day_slot is not None for p in picks)
        total_rounds = len(SURVIVOR_SLOTS) if has_day_slots else len(ROUND_ORDER)

    rounds_survived = 0
    for p in picks:
        if p.survived:
            rounds_survived += 1
        else:
            break

    prob = 1.0
    for p in picks[:rounds_survived + 1]:
        prob *= p.win_probability

    # Only truly survived if every round had a pick and all were correct
    survived_all = (
        not exhausted
        and rounds_survived == total_rounds
    )

    return SurvivorResult(
        year=year,
        strategy=strategy,
        picks=tuple(picks),
        survived_all=survived_all,
        rounds_survived=rounds_survived,
        total_rounds=total_rounds,
        survival_probability=prob,
        exhausted=exhausted,
    )


# ---------------------------------------------------------------------------
# Greedy strategy
# ---------------------------------------------------------------------------


def run_survivor_greedy(
    year: int,
    game_probs: pd.DataFrame,
) -> SurvivorResult:
    """Pick highest win-prob unused team each round/slot.

    When ``game_probs`` has a ``day_slot`` column, iterates over
    9 survivor slots instead of 6 rounds.

    Args:
        year: Tournament year.
        game_probs: DataFrame with YEAR, Team1, Seed1, Team2, Seed2,
                    CURRENT ROUND, Debiased_Prob, Team1_Win.
                    Optionally ``day_slot`` for 9-slot mode.
    """
    use_slots = "day_slot" in game_probs.columns and game_probs["day_slot"].notna().any()
    candidates = _build_round_candidates(game_probs, use_day_slots=use_slots)
    slot_order = list(SURVIVOR_SLOTS) if use_slots else list(ROUND_ORDER)

    used: set[str] = set()
    picks: list[SurvivorPick] = []
    exhausted = False

    for slot_name in slot_order:
        slot_cands = candidates.get(slot_name, [])
        available = [c for c in slot_cands if c["team"] not in used]

        if not available:
            logger.warning(
                "No available candidates for %s in %d", slot_name, year,
            )
            exhausted = True
            break

        best = max(available, key=lambda c: c["win_prob"])
        round_name = best.get("round_name", SLOT_TO_ROUND.get(slot_name, slot_name))
        pick = _make_pick(best, round_name, day_slot=slot_name if use_slots else None)
        picks.append(pick)
        used.add(pick.team)

        if not pick.survived:
            break

    return _result_from_picks(year, "greedy", picks, exhausted=exhausted)


# ---------------------------------------------------------------------------
# Optimal strategy (linear assignment)
# ---------------------------------------------------------------------------


def run_survivor_optimal(
    year: int,
    game_probs: pd.DataFrame,
) -> SurvivorResult:
    """Find the pick sequence that maximizes P(survive all slots).

    Solves a linear assignment problem on ``-log(win_prob)`` costs.
    Minimising the sum of log-costs is equivalent to maximising the
    product of win probabilities. Uses 9 survivor slots when day_slot
    data is available, otherwise falls back to 6 rounds.
    """
    use_slots = "day_slot" in game_probs.columns and game_probs["day_slot"].notna().any()
    candidates = _build_round_candidates(game_probs, use_day_slots=use_slots)
    slot_order = list(SURVIVOR_SLOTS) if use_slots else list(ROUND_ORDER)

    slots = [s for s in slot_order if s in candidates]

    if not slots:
        return _result_from_picks(year, "optimal", [], exhausted=True)

    cost, teams, slot_list = _build_cost_matrix(candidates, slots)
    assignments = _solve_assignment(cost, teams, slot_list)

    picks: list[SurvivorPick] = []
    for team_name, slot_name, _ in assignments:
        cand = next(
            c for c in candidates[slot_name] if c["team"] == team_name
        )
        round_name = cand.get("round_name", SLOT_TO_ROUND.get(slot_name, slot_name))
        picks.append(_make_pick(cand, round_name, day_slot=slot_name if use_slots else None))

    exhausted = len(assignments) < len(slot_order) or len(slots) < len(slot_order)
    return _result_from_picks(year, "optimal", picks, exhausted=exhausted)


# ---------------------------------------------------------------------------
# Monte Carlo survivor
# ---------------------------------------------------------------------------


def run_survivor_monte_carlo(
    year: int,
    game_probs: pd.DataFrame,
    strategy: str = "greedy",
    n_simulations: int = 1000,
    rng_seed: int = 42,
) -> SurvivorMonteCarloResult:
    """Run N survivor simulations with stochastic game outcomes.

    Model probabilities are fixed — each strategy makes the SAME picks
    every simulation. The randomness is in game outcomes: a 90% favorite
    still loses 10% of the time. We precompute each strategy's picks
    once, then check survival across N sampled outcome sets.

    This answers: "How robust is this strategy to outcome variance?"
    """
    candidates = _build_round_candidates(game_probs)
    rng = np.random.default_rng(rng_seed)

    # Precompute the strategy's picks (deterministic given model probs)
    if strategy == "optimal":
        base_result = run_survivor_optimal(year, game_probs)
    else:
        base_result = run_survivor_greedy(year, game_probs)

    # Determine slot order from the picks
    use_slots = any(p.day_slot is not None for p in base_result.picks)
    slot_order = list(SURVIVOR_SLOTS) if use_slots else list(ROUND_ORDER)

    # Build a lookup: slot_or_round -> (picked_team, win_prob)
    pick_plan: dict[str, tuple[str, float]] = {}
    for pick in base_result.picks:
        key = pick.day_slot if use_slots and pick.day_slot else pick.round_name
        pick_plan[key] = (pick.team, pick.win_probability)

    # Track which game each pick belongs to (for sampling outcomes)
    pick_games: dict[str, dict] = {}
    for pick in base_result.picks:
        key = pick.day_slot if use_slots and pick.day_slot else pick.round_name
        slot_cands = candidates.get(key, [])
        for cand in slot_cands:
            if cand["team"] == pick.team:
                pick_games[key] = cand
                break

    rounds_survived_counts: dict[int, int] = {}
    elimination_rounds: dict[str, int] = {}

    for _ in range(n_simulations):
        rounds_survived = 0

        for slot_name in slot_order:
            if slot_name not in pick_plan:
                break

            _, win_prob = pick_plan[slot_name]
            pick_wins = rng.random() < win_prob

            if pick_wins:
                rounds_survived += 1
            else:
                elimination_rounds[slot_name] = (
                    elimination_rounds.get(slot_name, 0) + 1
                )
                break

        rounds_survived_counts[rounds_survived] = (
            rounds_survived_counts.get(rounds_survived, 0) + 1
        )

    # Normalize
    n_picks = len(pick_plan)
    total_rounds = len(slot_order)
    survival_counts = {
        k: v / n_simulations
        for k, v in sorted(rounds_survived_counts.items())
    }
    survival_rate = (
        survival_counts.get(total_rounds, 0.0)
        if n_picks == total_rounds
        else 0.0
    )
    mean_survived = sum(
        k * v for k, v in rounds_survived_counts.items()
    ) / n_simulations

    # Team pick rates are deterministic (same picks every sim)
    team_rates: dict[str, dict[str, float]] = {}
    for pick in base_result.picks:
        key = pick.day_slot if use_slots and pick.day_slot else pick.round_name
        team_rates.setdefault(key, {})
        team_rates[key][pick.team] = 1.0

    # Sort by slot/round order
    slot_order_idx = {s: i for i, s in enumerate(slot_order)}
    elim_dist = {
        k: v / n_simulations
        for k, v in sorted(
            elimination_rounds.items(),
            key=lambda x: slot_order_idx.get(x[0], 99),
        )
    }

    logger.info(
        "MC survivor %d (%s, %d sims): survival_rate=%.1f%%, "
        "mean_rounds=%.1f",
        year, strategy, n_simulations,
        100.0 * survival_rate, mean_survived,
    )

    return SurvivorMonteCarloResult(
        year=year,
        strategy=f"{strategy}_mc",
        n_simulations=n_simulations,
        survival_rate=survival_rate,
        mean_rounds_survived=mean_survived,
        rounds_survived_distribution=survival_counts,
        team_pick_rates=team_rates,
        elimination_round_distribution=elim_dist,
    )


def _pick_greedy(
    round_cands: list[dict],
    used: set[str],
) -> dict | None:
    """Pick the highest win-prob unused team from candidates."""
    available = [c for c in round_cands if c["team"] not in used]
    if not available:
        return None
    return max(available, key=lambda c: c["win_prob"])


# ---------------------------------------------------------------------------
# Bracket-aware strategy
# ---------------------------------------------------------------------------


def _extract_bracket_candidates(
    bracket: Bracket,
    use_day_slots: bool = False,
) -> dict[str, list[dict]]:
    """Build per-slot candidate lists from a simulated bracket.

    Converts ``BracketGame`` objects into the same candidate dict
    format used by ``_build_round_candidates``, so the cost-matrix
    builder can consume them identically.

    The ``actual_winner`` field is left as the *predicted* winner
    (from the simulation). Callers that need real-world evaluation
    should replace this via ``_find_actual_result``.

    Args:
        bracket: A Bracket with simulated or actual games.
        use_day_slots: If True, key by ``game.day_slot`` instead
                       of ``game.round_name``.
    """
    candidates: dict[str, list[dict]] = {}

    for game in bracket.games:
        if game.winner is None:
            continue

        round_name = game.round_name
        if use_day_slots and game.day_slot is not None:
            key = game.day_slot
        else:
            key = round_name

        candidates.setdefault(key, [])

        prob = game.win_probability if game.win_probability is not None else 0.5

        base = {
            "round_name": round_name,
            "day_slot": game.day_slot,
        }

        # Team1 candidate
        candidates[key].append({
            **base,
            "team": game.team1.team,
            "seed": game.team1.seed,
            "opponent": game.team2.team,
            "opponent_seed": game.team2.seed,
            "win_prob": prob,
            "actual_winner": game.winner.team,
        })

        # Team2 candidate
        candidates[key].append({
            **base,
            "team": game.team2.team,
            "seed": game.team2.seed,
            "opponent": game.team1.team,
            "opponent_seed": game.team1.seed,
            "win_prob": 1.0 - prob,
            "actual_winner": game.winner.team,
        })

    return candidates


def run_survivor_bracket_aware(
    year: int,
    bracket: Bracket,
    actual_bracket: Bracket,
) -> SurvivorResult:
    """Derive survivor picks from a simulated bracket.

    Uses the simulation's predicted matchups and winners to build the
    candidate pool, then solves a bracket-constrained ILP to find the
    optimal pick set. The ILP ensures picks don't over-concentrate on
    one bracket side in regional slots, preserving viable options for
    F4/NCG.

    When bracket games have ``day_slot`` set, uses 9 survivor slots
    instead of 6 rounds. Otherwise falls back to the 6-round mode.

    After solving, each pick is evaluated against the actual bracket
    to determine whether it would have survived in reality.

    Args:
        year: Tournament year.
        bracket: Simulated bracket (from ``simulate_bracket_deterministic``).
        actual_bracket: Ground-truth bracket for survival evaluation.
    """
    use_slots = any(g.day_slot is not None for g in bracket.games)
    candidates = _extract_bracket_candidates(bracket, use_day_slots=use_slots)
    slot_order = list(SURVIVOR_SLOTS) if use_slots else list(ROUND_ORDER)

    slots = [s for s in slot_order if s in candidates]

    if not slots:
        return _result_from_picks(year, "bracket_aware", [], exhausted=True)

    cost, teams, slot_list = _build_cost_matrix(candidates, slots)
    team_side_map = build_team_side_map(bracket)
    assignments = solve_constrained_assignment(
        cost, teams, slot_list, team_side_map,
    )

    picks: list[SurvivorPick] = []
    for team_name, slot_name, win_prob in assignments:
        round_name = SLOT_TO_ROUND.get(slot_name, slot_name)
        # Check what actually happened to this team in this round
        actual = _find_actual_result(team_name, round_name, actual_bracket)

        if actual is not None:
            pick = SurvivorPick(
                round_name=round_name,
                team=team_name,
                seed=actual["seed"],
                opponent=actual["opponent"],
                opponent_seed=actual["opponent_seed"],
                win_probability=win_prob,
                actual_winner=actual["actual_winner"],
                survived=actual["actual_winner"] == team_name,
                day_slot=slot_name if use_slots else None,
            )
        else:
            sim_cand = next(
                c for c in candidates[slot_name] if c["team"] == team_name
            )
            pick = SurvivorPick(
                round_name=round_name,
                team=team_name,
                seed=sim_cand["seed"],
                opponent=sim_cand["opponent"],
                opponent_seed=sim_cand["opponent_seed"],
                win_probability=win_prob,
                actual_winner="N/A (eliminated earlier)",
                survived=False,
                day_slot=slot_name if use_slots else None,
            )
        picks.append(pick)

    exhausted = len(assignments) < len(slot_order)
    return _result_from_picks(year, "bracket_aware", picks, exhausted=exhausted)


# ---------------------------------------------------------------------------
# Monte Carlo optimal strategy
# ---------------------------------------------------------------------------


def run_survivor_mc_optimal(
    year: int,
    models: list,
    feature_lookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
    n_simulations: int = 1000,
    rng_seed: int = 42,
    feature_mode: str = "difference",
) -> SurvivorResult:
    """Find optimal survivor picks by averaging across simulated brackets.

    Runs *N* Monte Carlo bracket simulations. For each, records which
    teams reach which rounds and their win probabilities. Averages
    the ``-log(win_prob)`` costs across simulations to build a single
    aggregated cost matrix, then solves the linear assignment once.

    This handles uncertainty about who reaches each round: a team that
    reaches the NCG in 80% of sims will have a much lower average
    cost there than one that only makes it 10% of the time.

    Args:
        year: Tournament year.
        models: Trained ensemble models.
        feature_lookup: ``FeatureLookup`` for building feature vectors.
        matchups_df: Restructured matchups (for extracting R64).
        actual_bracket: Ground-truth bracket for survival evaluation.
        n_simulations: Number of bracket simulations to average over.
        rng_seed: Random seed for reproducibility.
        feature_mode: Feature construction mode.
    """
    from sports_quant.march_madness.simulate import (
        _build_day_slot_map,
        _extract_r64_matchups,
        _precompute_probabilities,
        _simulate_from_precomputed,
    )

    r64 = _extract_r64_matchups(matchups_df, year, feature_lookup)
    probabilities = _precompute_probabilities(
        year, r64, models, feature_lookup, feature_mode=feature_mode,
    )

    # Detect 9-slot mode from actual bracket
    day_slot_map = _build_day_slot_map(actual_bracket)
    use_slots = len(day_slot_map) > 0
    slot_order = list(SURVIVOR_SLOTS) if use_slots else list(ROUND_ORDER)

    # Collect all 64 team names
    all_teams: set[str] = set()
    for t1, t2 in r64:
        all_teams.add(t1.team)
        all_teams.add(t2.team)
    teams = sorted(all_teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    n_teams = len(teams)
    n_slots = len(slot_order)

    # Accumulators: sum of -log(p) and count of appearances
    cost_sum = np.zeros((n_teams, n_slots))
    cost_count = np.zeros((n_teams, n_slots))

    rng = np.random.default_rng(rng_seed)

    for _ in range(n_simulations):
        games = _simulate_from_precomputed(
            year, r64, probabilities, rng,
            day_slot_map=day_slot_map,
        )

        for game in games:
            if game.winner is None:
                continue

            col_key = game.day_slot if use_slots and game.day_slot else game.round_name
            if col_key not in slot_order:
                continue
            j = slot_order.index(col_key)

            prob = game.win_probability
            if prob is None:
                continue

            winner = game.winner.team
            if winner in team_idx:
                i = team_idx[winner]
                if game.team1.team == winner:
                    wp = max(prob, _PROB_FLOOR)
                else:
                    wp = max(1.0 - prob, _PROB_FLOOR)
                cost_sum[i, j] += -np.log(wp)
                cost_count[i, j] += 1

    # Build averaged cost matrix
    avg_cost = np.full((n_teams, n_slots), _INFEASIBLE)
    nonzero = cost_count > 0
    avg_cost[nonzero] = cost_sum[nonzero] / cost_count[nonzero]

    team_side_map = build_team_side_map_from_r64(r64)
    assignments = solve_constrained_assignment(
        avg_cost, teams, slot_order, team_side_map,
    )

    picks: list[SurvivorPick] = []
    for team_name, slot_name, win_prob in assignments:
        round_name = SLOT_TO_ROUND.get(slot_name, slot_name)
        actual = _find_actual_result(team_name, round_name, actual_bracket)

        if actual is not None:
            pick = SurvivorPick(
                round_name=round_name,
                team=team_name,
                seed=actual["seed"],
                opponent=actual["opponent"],
                opponent_seed=actual["opponent_seed"],
                win_probability=win_prob,
                actual_winner=actual["actual_winner"],
                survived=actual["actual_winner"] == team_name,
                day_slot=slot_name if use_slots else None,
            )
        else:
            pick = SurvivorPick(
                round_name=round_name,
                team=team_name,
                seed=0,
                opponent="unknown",
                opponent_seed=0,
                win_probability=win_prob,
                actual_winner="N/A (eliminated earlier)",
                survived=False,
                day_slot=slot_name if use_slots else None,
            )
        picks.append(pick)

    exhausted = len(assignments) < len(slot_order)

    logger.info(
        "MC optimal survivor %d (%d sims): %d picks, prob=%.4f",
        year, n_simulations, len(picks),
        np.prod([p.win_probability for p in picks]) if picks else 0.0,
    )

    return _result_from_picks(year, "mc_optimal", picks, exhausted=exhausted)


# ---------------------------------------------------------------------------
# Sequential MC optimal strategy (round-by-round re-optimization)
# ---------------------------------------------------------------------------


def _extract_round_matchups(
    actual_bracket: Bracket,
    round_name: str,
    feature_lookup,
    year: int,
) -> list[tuple]:
    """Extract actual matchups for a round as TeamStats pairs.

    Reads the actual bracket to find the games in ``round_name``,
    then converts each team into a ``TeamStats`` via ``feature_lookup``.
    Games are returned sorted by ``game_index`` so adjacent winners
    pair correctly for the next round.
    """
    from sports_quant.march_madness._feature_builder import TeamStats

    round_games = sorted(
        [g for g in actual_bracket.games if g.round_name == round_name],
        key=lambda g: g.game_index,
    )

    matchups: list[tuple] = []
    for game in round_games:
        t1 = feature_lookup.get_team(game.team1.team, year, game.team1.seed)
        t2 = feature_lookup.get_team(game.team2.team, year, game.team2.seed)
        matchups.append((t1, t2))

    return matchups


def _extract_round_winners(
    actual_bracket: Bracket,
    round_name: str,
    feature_lookup,
    year: int,
) -> list:
    """Get winners of a round as TeamStats, sorted by game_index.

    Used to construct matchups for the next round: adjacent winners
    are paired (winner of game 0 vs winner of game 1, etc.).
    """
    round_games = sorted(
        [g for g in actual_bracket.games if g.round_name == round_name],
        key=lambda g: g.game_index,
    )

    winners = []
    for game in round_games:
        if game.winner is None:
            continue
        w = game.winner
        winners.append(feature_lookup.get_team(w.team, year, w.seed))

    return winners


def run_survivor_mc_optimal_sequential(
    year: int,
    models: list,
    feature_lookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
    n_simulations: int = 1000,
    rng_seed: int = 42,
    feature_mode: str = "difference",
) -> SurvivorResult:
    """Sequential MC optimal: re-optimize after each round's results.

    Mirrors how a real survivor pool works. Before each round:

    1. Use the **actual** matchups for the current round (known after
       the previous round completes).
    2. Run N MC simulations from the current round **forward** to NCG.
    3. Build a cost matrix for remaining rounds, excluding used teams.
    4. Solve the assignment — commit only the current round's pick.
    5. Observe the actual result (survived or eliminated).
    6. Repeat with updated information for the next round.

    This gives the optimizer progressively more information as the
    tournament unfolds, unlike ``run_survivor_mc_optimal`` which
    commits all 6 picks before R64.

    Args:
        year: Tournament year.
        models: Trained ensemble models.
        feature_lookup: ``FeatureLookup`` for building feature vectors.
        matchups_df: Restructured matchups (for extracting R64).
        actual_bracket: Ground-truth bracket for results and matchups.
        n_simulations: Number of forward simulations per round.
        rng_seed: Random seed for reproducibility.
        feature_mode: Feature construction mode.
    """
    from sports_quant.march_madness.simulate import (
        _build_day_slot_map,
        _extract_r64_matchups,
        _precompute_probabilities,
        _simulate_from_round,
    )

    # Precompute all pairwise probabilities once (they don't change)
    r64 = _extract_r64_matchups(matchups_df, year, feature_lookup)
    probabilities = _precompute_probabilities(
        year, r64, models, feature_lookup, feature_mode=feature_mode,
    )

    # Detect 9-slot mode from actual bracket
    day_slot_map = _build_day_slot_map(actual_bracket)
    use_slots = len(day_slot_map) > 0
    slot_order = list(SURVIVOR_SLOTS) if use_slots else list(ROUND_ORDER)

    # Collect all 64 team names
    all_teams: set[str] = set()
    for t1, t2 in r64:
        all_teams.add(t1.team)
        all_teams.add(t2.team)
    teams = sorted(all_teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    team_side_map = build_team_side_map_from_r64(r64)
    used_teams: set[str] = set()
    side_counts: dict[str, int] = {"left": 0, "right": 0}
    picks: list[SurvivorPick] = []
    exhausted = False

    # Track which round's matchups we've already built so we don't
    # re-extract when consecutive slots share a round (e.g. R64_D1, R64_D2).
    last_round_idx: int | None = None
    current_matchups: list[tuple] = []

    for slot_idx, slot_name in enumerate(slot_order):
        round_name = SLOT_TO_ROUND.get(slot_name, slot_name)
        round_idx = ROUND_ORDER.index(round_name)

        # Build matchups for this round (only when the round changes)
        if round_idx != last_round_idx:
            if round_idx == 0:
                current_matchups = list(r64)
            else:
                prev_round = ROUND_ORDER[round_idx - 1]
                prev_winners = _extract_round_winners(
                    actual_bracket, prev_round, feature_lookup, year,
                )
                if len(prev_winners) < 2:
                    exhausted = True
                    break
                current_matchups = [
                    (prev_winners[i], prev_winners[i + 1])
                    for i in range(0, len(prev_winners), 2)
                ]
            last_round_idx = round_idx

        remaining_slots = list(slot_order[slot_idx:])
        n_remaining = len(remaining_slots)
        n_teams_total = len(teams)

        # Run N MC sims from this round forward, accumulate costs
        cost_sum = np.zeros((n_teams_total, n_remaining))
        cost_count = np.zeros((n_teams_total, n_remaining))

        rng = np.random.default_rng(rng_seed + slot_idx * 1000)

        for _ in range(n_simulations):
            games = _simulate_from_round(
                year, round_idx, current_matchups, probabilities, rng,
                day_slot_map=day_slot_map,
            )

            for game in games:
                if game.winner is None or game.win_probability is None:
                    continue

                col_key = (
                    game.day_slot
                    if use_slots and game.day_slot
                    else game.round_name
                )
                if col_key not in remaining_slots:
                    continue
                j = remaining_slots.index(col_key)

                winner = game.winner.team
                if winner not in team_idx:
                    continue

                i = team_idx[winner]
                if game.team1.team == winner:
                    wp = max(game.win_probability, _PROB_FLOOR)
                else:
                    wp = max(1.0 - game.win_probability, _PROB_FLOOR)
                cost_sum[i, j] += -np.log(wp)
                cost_count[i, j] += 1

        # Build averaged cost matrix, excluding used teams
        avg_cost = np.full((n_teams_total, n_remaining), _INFEASIBLE)
        nonzero = cost_count > 0
        avg_cost[nonzero] = cost_sum[nonzero] / cost_count[nonzero]

        for used_team in used_teams:
            if used_team in team_idx:
                avg_cost[team_idx[used_team], :] = _INFEASIBLE

        # Solve assignment with bracket-side constraints
        assignments = solve_constrained_assignment(
            avg_cost, teams, remaining_slots, team_side_map,
            prior_side_counts=side_counts,
        )

        if not assignments:
            exhausted = True
            break

        # Commit only the CURRENT slot's pick
        current_pick = next(
            (a for a in assignments if a[1] == slot_name), None,
        )
        if current_pick is None:
            exhausted = True
            break

        team_name, _, win_prob = current_pick
        used_teams.add(team_name)

        # Update side counts for sequential constraint tracking
        if round_name in REGIONAL_ROUNDS:
            pick_side = team_side_map.get(team_name)
            if pick_side:
                side_counts = {
                    **side_counts,
                    pick_side: side_counts[pick_side] + 1,
                }

        # Check actual result
        actual = _find_actual_result(team_name, round_name, actual_bracket)
        if actual is not None:
            pick = SurvivorPick(
                round_name=round_name,
                team=team_name,
                seed=actual["seed"],
                opponent=actual["opponent"],
                opponent_seed=actual["opponent_seed"],
                win_probability=win_prob,
                actual_winner=actual["actual_winner"],
                survived=actual["actual_winner"] == team_name,
                day_slot=slot_name if use_slots else None,
            )
        else:
            pick = SurvivorPick(
                round_name=round_name,
                team=team_name,
                seed=0,
                opponent="unknown",
                opponent_seed=0,
                win_probability=win_prob,
                actual_winner="N/A (eliminated earlier)",
                survived=False,
                day_slot=slot_name if use_slots else None,
            )

        picks.append(pick)

        if not pick.survived:
            break

    logger.info(
        "MC optimal sequential %d (%d sims/slot): %d picks, survived %s",
        year, n_simulations, len(picks),
        "/".join("OK" if p.survived else "XX" for p in picks),
    )

    return _result_from_picks(
        year, "mc_optimal_seq", picks, exhausted=exhausted,
    )


# ---------------------------------------------------------------------------
# Live prediction mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSurvivorState:
    """State of a survivor pool in progress.

    ``completed_rounds`` can contain either round names (R64, R32, ...)
    or survivor slot names (R64_D1, R64_D2, ...) depending on mode.
    """

    year: int
    completed_rounds: tuple[str, ...]
    picks_made: tuple[SurvivorPick, ...]
    teams_used: frozenset[str]
    still_alive: bool


def run_survivor_live(
    state: LiveSurvivorState,
    models: list,
    feature_lookup,
    matchups_df: pd.DataFrame,
    known_results: pd.DataFrame,
    n_simulations: int = 1000,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Recommend a survivor pick for the next round of a live tournament.

    Evaluates each candidate pick for the next round by running MC
    simulations of all remaining rounds using known matchups and
    model probabilities.

    Note: This works best when matchups for remaining rounds are known
    (e.g., picking for NCG when F4 results are in). For earlier picks
    where future matchups are unknown, future integration with
    simulate.py's forward simulation will generate hypothetical
    matchups. Currently, future rounds with no data are skipped.

    Args:
        state: Current survivor pool state.
        models: Trained models (reserved for future forward sim integration).
        feature_lookup: FeatureLookup (reserved for future forward sim).
        matchups_df: Known matchups.
        known_results: Results with Debiased_Prob and CURRENT ROUND.
        n_simulations: MC simulations per candidate.
        rng_seed: Random seed.

    Returns:
        Dict of team_name -> P(survive all remaining rounds with known matchups).
    """
    rng = np.random.default_rng(rng_seed)

    # Determine whether we're in slot mode or round mode
    use_slots = (
        "day_slot" in known_results.columns
        and known_results["day_slot"].notna().any()
    )
    slot_order = list(SURVIVOR_SLOTS) if use_slots else list(ROUND_ORDER)

    # Figure out which slot/round to pick for
    remaining = [
        s for s in slot_order if s not in state.completed_rounds
    ]
    if not remaining:
        return {}

    next_slot = remaining[0]
    future_slots = remaining[1:]

    # Map slot to round number for filtering
    round_name_for_slot = SLOT_TO_ROUND.get(next_slot, next_slot)
    round_num_lookup = {v: k for k, v in ROUND_NAMES.items()}

    # Build candidates for the next slot from known results
    next_round_num = round_num_lookup.get(round_name_for_slot)
    if next_round_num is None:
        return {}

    filtered = known_results[known_results["CURRENT ROUND"] == next_round_num]
    next_round_candidates = _build_round_candidates(
        filtered, use_day_slots=use_slots,
    ).get(next_slot, [])

    available = [
        c for c in next_round_candidates
        if c["team"] not in state.teams_used
    ]

    if not available:
        return {}

    # For each candidate, simulate remaining tournament
    team_survival: dict[str, float] = {}

    for candidate in available:
        survive_count = 0

        for sim_idx in range(n_simulations):
            sim_rng = np.random.default_rng(rng_seed + sim_idx)

            # Does our pick survive this round?
            pick_wins = sim_rng.random() < candidate["win_prob"]
            if not pick_wins:
                continue

            # Simulate future slots with greedy strategy
            survived_future = True
            used = set(state.teams_used) | {candidate["team"]}

            for future_slot in future_slots:
                future_round = SLOT_TO_ROUND.get(future_slot, future_slot)
                round_num = round_num_lookup.get(future_round)
                if round_num is None:
                    break
                future_cands = _build_round_candidates(
                    known_results[
                        known_results["CURRENT ROUND"] == round_num
                    ],
                    use_day_slots=use_slots,
                ).get(future_slot, [])

                if not future_cands:
                    # No known matchups for future rounds — can't evaluate
                    break

                # Simulate outcomes for this round
                simulated_cands: list[dict] = []
                for i in range(0, len(future_cands), 2):
                    c1 = future_cands[i]
                    c2 = future_cands[i + 1]
                    t1_wins = sim_rng.random() < c1["win_prob"]
                    winner = c1["team"] if t1_wins else c2["team"]
                    simulated_cands.append(
                        {**c1, "actual_winner": winner},
                    )
                    simulated_cands.append(
                        {**c2, "actual_winner": winner},
                    )

                # Greedy pick
                pick = _pick_greedy(simulated_cands, used)
                if pick is None:
                    survived_future = False
                    break

                used.add(pick["team"])
                if pick["actual_winner"] != pick["team"]:
                    survived_future = False
                    break

            if survived_future:
                survive_count += 1

        team_survival[candidate["team"]] = survive_count / n_simulations

    # Sort descending
    return dict(
        sorted(team_survival.items(), key=lambda x: x[1], reverse=True)
    )


# ---------------------------------------------------------------------------
# Utility: merge round info into debiased results
# ---------------------------------------------------------------------------


def prepare_game_probs(
    debiased_df: pd.DataFrame,
    matchups_df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Merge CURRENT ROUND and day_slot from matchups into debiased results.

    The debiased_results.csv lacks round info. This joins it back
    from restructured_matchups.csv by matching on team names and year.
    When matchups_df has a ``day_slot`` column, it is carried through.
    """
    yr_matchups = matchups_df[matchups_df["YEAR"] == year].reset_index(
        drop=True,
    )
    yr_debiased = debiased_df[debiased_df["YEAR"] == year].reset_index(
        drop=True,
    )

    if len(yr_debiased) != len(yr_matchups):
        raise ValueError(
            f"Row count mismatch: {len(yr_debiased)} debiased vs "
            f"{len(yr_matchups)} matchups for year {year}"
        )

    result = yr_debiased.copy()
    result["CURRENT ROUND"] = yr_matchups["CURRENT ROUND"].values

    # Carry day_slot through if present
    if "day_slot" in yr_matchups.columns:
        result["day_slot"] = yr_matchups["day_slot"].values

    return result
