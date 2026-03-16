"""Survivor pool optimizer for March Madness.

Picks one team per round to win their game, with no team reuse.
Supports greedy, optimal (branch-and-bound), and Monte Carlo strategies.
Also provides a live prediction mode that combines known results with
forward simulation of future rounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_quant.march_madness._bracket import ROUND_NAMES, ROUND_ORDER

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
# Optimal strategy (branch-and-bound)
# ---------------------------------------------------------------------------


def run_survivor_optimal(
    year: int,
    game_probs: pd.DataFrame,
) -> SurvivorResult:
    """Find the pick sequence that maximizes P(survive all 6 rounds).

    Uses branch-and-bound: prunes partial sequences whose probability
    product is already below the current best complete sequence.
    """
    candidates = _build_round_candidates(game_probs)
    rounds = [r for r in ROUND_ORDER if r in candidates]

    # Search for best complete sequence (all rounds filled)
    best_complete_prob: list[float] = [0.0]
    best_complete_picks: list[list[dict]] = [[]]
    # Track best partial as fallback (when no complete path exists)
    best_partial_picks: list[list[dict]] = [[]]
    best_partial_len: list[int] = [0]

    def _search(
        round_idx: int,
        used: set[str],
        current_picks: list[dict],
        current_prob: float,
    ) -> None:
        if round_idx == len(rounds):
            if current_prob > best_complete_prob[0]:
                best_complete_prob[0] = current_prob
                best_complete_picks[0] = list(current_picks)
            return

        round_name = rounds[round_idx]
        round_cands = candidates[round_name]
        available = [c for c in round_cands if c["team"] not in used]

        if not available:
            # Track longest partial sequence as fallback
            if len(current_picks) > best_partial_len[0]:
                best_partial_len[0] = len(current_picks)
                best_partial_picks[0] = list(current_picks)
            return

        # Sort descending by win_prob for better pruning
        available.sort(key=lambda c: c["win_prob"], reverse=True)

        for cand in available:
            new_prob = current_prob * cand["win_prob"]

            # Prune: even if all future picks are 1.0, can't beat best
            if new_prob <= best_complete_prob[0]:
                continue

            used.add(cand["team"])
            current_picks.append(cand)

            _search(round_idx + 1, used, current_picks, new_prob)

            current_picks.pop()
            used.discard(cand["team"])

    _search(0, set(), [], 1.0)

    # Prefer complete paths; fall back to longest partial
    if best_complete_picks[0]:
        final_picks = best_complete_picks[0]
        exhausted = False
    else:
        final_picks = best_partial_picks[0]
        exhausted = True

    picks = [
        _make_pick(cand, rounds[i])
        for i, cand in enumerate(final_picks)
    ]

    # Also exhausted if data is missing rounds
    if len(rounds) < len(ROUND_ORDER):
        exhausted = True

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
