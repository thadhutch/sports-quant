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
    ROUND_NAMES,
    ROUND_ORDER,
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
) -> dict[str, list[dict]]:
    """Build per-round candidate lists from game probabilities.

    Each candidate is a dict with team info and win probability for
    both the Team1 side and Team2 side of each game.

    Args:
        game_probs: DataFrame with YEAR, Team1, Seed1, Team2, Seed2,
                    CURRENT ROUND, Debiased_Prob, Team1_Win.

    Returns:
        round_name -> list of candidate dicts, each with keys:
        team, seed, opponent, opponent_seed, win_prob, actual_winner.
    """
    candidates: dict[str, list[dict]] = {}

    for _, row in game_probs.iterrows():
        round_num = int(row["CURRENT ROUND"])
        round_name = ROUND_NAMES.get(round_num)
        if round_name is None:
            continue

        prob = float(row["Debiased_Prob"])
        actual_winner = (
            row["Team1"] if int(row["Team1_Win"]) == 1 else row["Team2"]
        )

        candidates.setdefault(round_name, [])

        # Team1 candidate
        candidates[round_name].append({
            "team": row["Team1"],
            "seed": int(row["Seed1"]),
            "opponent": row["Team2"],
            "opponent_seed": int(row["Seed2"]),
            "win_prob": prob,
            "actual_winner": actual_winner,
        })

        # Team2 candidate
        candidates[round_name].append({
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


def _make_pick(candidate: dict, round_name: str) -> SurvivorPick:
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
    )


def _result_from_picks(
    year: int,
    strategy: str,
    picks: list[SurvivorPick],
    total_rounds: int = len(ROUND_ORDER),
    exhausted: bool = False,
) -> SurvivorResult:
    """Build a SurvivorResult from a list of picks."""
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
    """Pick highest win-prob unused team each round.

    Args:
        year: Tournament year.
        game_probs: DataFrame with YEAR, Team1, Seed1, Team2, Seed2,
                    CURRENT ROUND, Debiased_Prob, Team1_Win.
    """
    candidates = _build_round_candidates(game_probs)
    used: set[str] = set()
    picks: list[SurvivorPick] = []
    exhausted = False

    for round_name in ROUND_ORDER:
        round_cands = candidates.get(round_name, [])
        available = [c for c in round_cands if c["team"] not in used]

        if not available:
            logger.warning(
                "No available candidates for %s in %d", round_name, year,
            )
            exhausted = True
            break

        best = max(available, key=lambda c: c["win_prob"])
        pick = _make_pick(best, round_name)
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
    """Find the pick sequence that maximizes P(survive all 6 rounds).

    Solves a linear assignment problem on ``-log(win_prob)`` costs.
    Minimising the sum of log-costs is equivalent to maximising the
    product of win probabilities.
    """
    candidates = _build_round_candidates(game_probs)
    rounds = [r for r in ROUND_ORDER if r in candidates]

    if not rounds:
        return _result_from_picks(year, "optimal", [], exhausted=True)

    cost, teams, round_list = _build_cost_matrix(candidates, rounds)
    assignments = _solve_assignment(cost, teams, round_list)

    picks: list[SurvivorPick] = []
    for team_name, round_name, _ in assignments:
        # Look up the full candidate dict for opponent/seed/actual_winner
        cand = next(
            c for c in candidates[round_name] if c["team"] == team_name
        )
        picks.append(_make_pick(cand, round_name))

    exhausted = len(assignments) < len(ROUND_ORDER) or len(rounds) < len(ROUND_ORDER)
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

    # Build a lookup: round_name -> (picked_team, win_prob)
    pick_plan: dict[str, tuple[str, float]] = {}
    for pick in base_result.picks:
        pick_plan[pick.round_name] = (pick.team, pick.win_probability)

    # Track which game each pick belongs to (for sampling outcomes)
    # We need the team's opponent to know which game to sample
    pick_games: dict[str, dict] = {}
    for pick in base_result.picks:
        # Find this pick's game in candidates
        round_cands = candidates.get(pick.round_name, [])
        for cand in round_cands:
            if cand["team"] == pick.team:
                pick_games[pick.round_name] = cand
                break

    rounds_survived_counts: dict[int, int] = {}
    elimination_rounds: dict[str, int] = {}

    for _ in range(n_simulations):
        rounds_survived = 0

        for round_name in ROUND_ORDER:
            if round_name not in pick_plan:
                break

            _, win_prob = pick_plan[round_name]
            pick_wins = rng.random() < win_prob

            if pick_wins:
                rounds_survived += 1
            else:
                elimination_rounds[round_name] = (
                    elimination_rounds.get(round_name, 0) + 1
                )
                break

        rounds_survived_counts[rounds_survived] = (
            rounds_survived_counts.get(rounds_survived, 0) + 1
        )

    # Normalize
    n_picks = len(pick_plan)
    total_rounds = len(ROUND_ORDER)
    survival_counts = {
        k: v / n_simulations
        for k, v in sorted(rounds_survived_counts.items())
    }
    # Survival means surviving all 6 rounds. If the strategy doesn't
    # have 6 picks (e.g., greedy ran out of candidates), it can never
    # achieve full survival.
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
        team_rates.setdefault(pick.round_name, {})
        team_rates[pick.round_name][pick.team] = 1.0

    elim_dist = {
        k: v / n_simulations
        for k, v in sorted(
            elimination_rounds.items(),
            key=lambda x: ROUND_ORDER.index(x[0])
            if x[0] in ROUND_ORDER else 99,
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
) -> dict[str, list[dict]]:
    """Build per-round candidate lists from a simulated bracket.

    Converts ``BracketGame`` objects into the same candidate dict
    format used by ``_build_round_candidates``, so the cost-matrix
    builder can consume them identically.

    The ``actual_winner`` field is left as the *predicted* winner
    (from the simulation). Callers that need real-world evaluation
    should replace this via ``_find_actual_result``.
    """
    candidates: dict[str, list[dict]] = {}

    for game in bracket.games:
        if game.winner is None:
            continue

        round_name = game.round_name
        candidates.setdefault(round_name, [])

        prob = game.win_probability if game.win_probability is not None else 0.5

        # Team1 candidate
        candidates[round_name].append({
            "team": game.team1.team,
            "seed": game.team1.seed,
            "opponent": game.team2.team,
            "opponent_seed": game.team2.seed,
            "win_prob": prob,
            "actual_winner": game.winner.team,
        })

        # Team2 candidate
        candidates[round_name].append({
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
    candidate pool, then solves the linear assignment to find the
    optimal pick set. Teams predicted to go deep are naturally
    available in later rounds, so the optimizer reserves them.

    After solving, each pick is evaluated against the actual bracket
    to determine whether it would have survived in reality.

    Args:
        year: Tournament year.
        bracket: Simulated bracket (from ``simulate_bracket_deterministic``).
        actual_bracket: Ground-truth bracket for survival evaluation.
    """
    candidates = _extract_bracket_candidates(bracket)
    rounds = [r for r in ROUND_ORDER if r in candidates]

    if not rounds:
        return _result_from_picks(year, "bracket_aware", [], exhausted=True)

    cost, teams, round_list = _build_cost_matrix(candidates, rounds)
    assignments = _solve_assignment(cost, teams, round_list)

    picks: list[SurvivorPick] = []
    for team_name, round_name, win_prob in assignments:
        # Check what actually happened to this team in this round
        actual = _find_actual_result(team_name, round_name, actual_bracket)

        if actual is not None:
            # Team played in this round — use real result
            pick = SurvivorPick(
                round_name=round_name,
                team=team_name,
                seed=actual["seed"],
                opponent=actual["opponent"],
                opponent_seed=actual["opponent_seed"],
                win_probability=win_prob,
                actual_winner=actual["actual_winner"],
                survived=actual["actual_winner"] == team_name,
            )
        else:
            # Team was eliminated before reaching this round
            sim_cand = next(
                c for c in candidates[round_name] if c["team"] == team_name
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
            )
        picks.append(pick)

    exhausted = len(assignments) < len(ROUND_ORDER)
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
        _extract_r64_matchups,
        _precompute_probabilities,
        _simulate_from_precomputed,
    )

    r64 = _extract_r64_matchups(matchups_df, year, feature_lookup)
    probabilities = _precompute_probabilities(
        year, r64, models, feature_lookup, feature_mode=feature_mode,
    )

    # Collect all 64 team names
    all_teams: set[str] = set()
    for t1, t2 in r64:
        all_teams.add(t1.team)
        all_teams.add(t2.team)
    teams = sorted(all_teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    rounds = list(ROUND_ORDER)
    n_teams = len(teams)
    n_rounds = len(rounds)

    # Accumulators: sum of -log(p) and count of appearances
    cost_sum = np.zeros((n_teams, n_rounds))
    cost_count = np.zeros((n_teams, n_rounds))

    rng = np.random.default_rng(rng_seed)

    for _ in range(n_simulations):
        games = _simulate_from_precomputed(year, r64, probabilities, rng)

        for game in games:
            if game.winner is None:
                continue

            j = rounds.index(game.round_name)
            prob = game.win_probability
            if prob is None:
                continue

            # Winner's probability
            winner = game.winner.team
            if winner in team_idx:
                i = team_idx[winner]
                # Determine actual win probability for the winner
                if game.team1.team == winner:
                    wp = max(prob, _PROB_FLOOR)
                else:
                    wp = max(1.0 - prob, _PROB_FLOOR)
                cost_sum[i, j] += -np.log(wp)
                cost_count[i, j] += 1

    # Build averaged cost matrix
    avg_cost = np.full((n_teams, n_rounds), _INFEASIBLE)
    nonzero = cost_count > 0
    avg_cost[nonzero] = cost_sum[nonzero] / cost_count[nonzero]

    assignments = _solve_assignment(avg_cost, teams, rounds)

    picks: list[SurvivorPick] = []
    for team_name, round_name, win_prob in assignments:
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
            )
        picks.append(pick)

    exhausted = len(assignments) < len(ROUND_ORDER)

    logger.info(
        "MC optimal survivor %d (%d sims): %d picks, prob=%.4f",
        year, n_simulations, len(picks),
        np.prod([p.win_probability for p in picks]) if picks else 0.0,
    )

    return _result_from_picks(year, "mc_optimal", picks, exhausted=exhausted)


# ---------------------------------------------------------------------------
# Live prediction mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveSurvivorState:
    """State of a survivor pool in progress."""

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

    # Figure out which round to pick for
    remaining_rounds = [
        r for r in ROUND_ORDER if r not in state.completed_rounds
    ]
    if not remaining_rounds:
        return {}

    next_round = remaining_rounds[0]
    future_rounds = remaining_rounds[1:]

    # Build candidates for the next round from known results
    next_round_candidates = _build_round_candidates(
        known_results[
            known_results["CURRENT ROUND"]
            == {v: k for k, v in ROUND_NAMES.items()}[next_round]
        ],
    ).get(next_round, [])

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

            # Simulate future rounds with greedy strategy
            survived_future = True
            used = set(state.teams_used) | {candidate["team"]}

            for future_round in future_rounds:
                round_num = {v: k for k, v in ROUND_NAMES.items()}[
                    future_round
                ]
                future_cands = _build_round_candidates(
                    known_results[
                        known_results["CURRENT ROUND"] == round_num
                    ],
                ).get(future_round, [])

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
    """Merge CURRENT ROUND from matchups into debiased results.

    The debiased_results.csv lacks round info. This joins it back
    from restructured_matchups.csv by matching on team names and year.
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
    return result
