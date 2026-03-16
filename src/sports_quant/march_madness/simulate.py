"""Forward bracket simulation engine.

Simulates a full 63-game tournament bracket by predicting each game
from R64 onward, advancing winners round by round. Supports both
deterministic (best-guess) and Monte Carlo (sampled) modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_quant.march_madness._bracket import (
    ROUND_ORDER,
    Bracket,
    BracketGame,
    BracketSlot,
    determine_upset,
)
from sports_quant.march_madness._bracket_builder import (
    _CANONICAL_SEED_PAIR_POS,
    _assign_region,
    _bracket_tree_order,
)
from sports_quant.march_madness._debiasing import (
    swap_difference_features,
    swap_team_columns,
)
from sports_quant.march_madness._feature_builder import (
    FeatureLookup,
    TeamStats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Result of a single deterministic bracket simulation."""

    year: int
    bracket: Bracket
    accuracy_by_round: dict[str, tuple[int, int]]
    overall_accuracy: tuple[int, int]


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregated results from N Monte Carlo bracket simulations."""

    year: int
    n_simulations: int
    team_round_rates: dict[str, dict[str, float]]
    mean_accuracy_by_round: dict[str, float]
    champion_distribution: dict[str, float]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def _predict_game(
    team1: TeamStats,
    team2: TeamStats,
    models: list,
    feature_lookup: FeatureLookup,
    feature_mode: str = "difference",
) -> float:
    """Predict Team1 win probability with debiasing.

    Builds feature vectors for original and swapped team orderings,
    averages probabilities across all models and both orderings.

    Args:
        team1: First team.
        team2: Second team.
        models: Trained ensemble models.
        feature_lookup: For building feature vectors.
        feature_mode: "difference" or "raw".

    Returns:
        Debiased probability that team1 wins.
    """
    if feature_mode == "combined":
        features = feature_lookup.build_combined_difference_features(team1, team2)
        features_swapped = swap_difference_features(features)
    elif feature_mode == "difference":
        features = feature_lookup.build_difference_features(team1, team2)
        features_swapped = swap_difference_features(features)
    else:
        features = feature_lookup.build_matchup_features(team1, team2)
        features_swapped = swap_team_columns(features)

    original_probs = [
        m.predict_proba(features)[:, 1][0] for m in models
    ]
    swapped_probs = [
        1.0 - m.predict_proba(features_swapped)[:, 1][0] for m in models
    ]

    avg_original = np.mean(original_probs)
    avg_swapped = np.mean(swapped_probs)
    return float((avg_original + avg_swapped) / 2.0)


# ---------------------------------------------------------------------------
# R64 extraction
# ---------------------------------------------------------------------------


def _extract_r64_matchups(
    matchups_df: pd.DataFrame,
    year: int,
    feature_lookup: FeatureLookup,
) -> list[tuple[TeamStats, TeamStats]]:
    """Extract the 32 Round-of-64 matchups in canonical bracket order.

    Uses the bracket-tree ordering from actual results to get correct
    region grouping, then returns TeamStats pairs for each game.
    """
    year_df = matchups_df[matchups_df["YEAR"] == year].copy()
    r64_df = year_df[year_df["CURRENT ROUND"] == 64]

    if len(r64_df) != 32:
        raise ValueError(
            f"Expected 32 R64 games for {year}, got {len(r64_df)}"
        )

    # Use bracket-tree ordering to get canonical region grouping
    ordered = _bracket_tree_order(year_df)
    if ordered is None:
        raise ValueError(
            f"Cannot reconstruct bracket tree for {year}. "
            f"Incomplete matchup data."
        )

    matchups: list[tuple[TeamStats, TeamStats]] = []
    for idx in ordered["R64"]:
        row = year_df.loc[idx]
        t1 = feature_lookup.get_team(
            row["Team1"], year, int(row["Seed1"]),
        )
        t2 = feature_lookup.get_team(
            row["Team2"], year, int(row["Seed2"]),
        )
        matchups.append((t1, t2))

    return matchups


# ---------------------------------------------------------------------------
# Single bracket simulation (shared by deterministic and MC)
# ---------------------------------------------------------------------------


def _simulate_single_bracket(
    year: int,
    r64_matchups: list[tuple[TeamStats, TeamStats]],
    models: list,
    feature_lookup: FeatureLookup,
    rng: np.random.Generator | None,
    feature_mode: str = "difference",
) -> tuple[list[BracketGame], dict[str, list[float]]]:
    """Simulate one complete bracket from R64 through NCG.

    Args:
        year: Tournament year.
        r64_matchups: The 32 R64 matchups in canonical bracket order.
        models: Trained ensemble models.
        feature_lookup: For building feature vectors.
        rng: If provided, sample outcomes from probabilities (MC mode).
             If None, always pick the higher-probability team (deterministic).
        feature_mode: "difference" or "raw".

    Returns:
        Tuple of (list of 63 BracketGames, dict of round -> probabilities).
    """
    games: list[BracketGame] = []
    round_probs: dict[str, list[float]] = {}
    current_matchups = list(r64_matchups)

    for round_name in ROUND_ORDER:
        round_probs[round_name] = []
        winners: list[TeamStats] = []

        for game_idx, (t1, t2) in enumerate(current_matchups):
            prob = _predict_game(
                t1, t2, models, feature_lookup, feature_mode,
            )
            round_probs[round_name].append(prob)

            if rng is not None:
                team1_wins = rng.random() < prob
            else:
                team1_wins = prob > 0.5

            winner_stats = t1 if team1_wins else t2
            winners.append(winner_stats)

            slot1 = BracketSlot(team=t1.team, seed=t1.seed)
            slot2 = BracketSlot(team=t2.team, seed=t2.seed)
            winner_slot = slot1 if team1_wins else slot2

            game = BracketGame(
                round_name=round_name,
                region=_assign_region(round_name, game_idx),
                game_index=game_idx,
                team1=slot1,
                team2=slot2,
                winner=winner_slot,
                win_probability=prob,
                is_upset=determine_upset(t1.seed, t2.seed, team1_wins),
                is_correct=None,  # set later during evaluation
            )
            games.append(game)

        # Pair adjacent winners for next round (skip after NCG)
        if len(winners) >= 2:
            current_matchups = [
                (winners[i], winners[i + 1])
                for i in range(0, len(winners), 2)
            ]

    return games, round_probs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _build_actual_winners(actual: Bracket) -> dict[tuple[str, int], str]:
    """Build lookup of (round_name, game_index) -> actual winner name."""
    return {
        (g.round_name, g.game_index): g.winner.team
        for g in actual.games
        if g.winner is not None
    }


def _evaluate_games(
    games: list[BracketGame],
    actual_winners: dict[tuple[str, int], str],
) -> list[BracketGame]:
    """Set is_correct on each game by comparing to actual winners."""
    evaluated: list[BracketGame] = []
    for g in games:
        actual = actual_winners.get((g.round_name, g.game_index))
        is_correct = None
        if actual is not None and g.winner is not None:
            is_correct = g.winner.team == actual
        evaluated.append(
            BracketGame(
                round_name=g.round_name,
                region=g.region,
                game_index=g.game_index,
                team1=g.team1,
                team2=g.team2,
                winner=g.winner,
                win_probability=g.win_probability,
                is_upset=g.is_upset,
                is_correct=is_correct,
            )
        )
    return evaluated


def _accuracy_by_round(
    games: list[BracketGame],
) -> dict[str, tuple[int, int]]:
    """Compute (correct, total) per round from evaluated games."""
    result: dict[str, tuple[int, int]] = {}
    for round_name in ROUND_ORDER:
        round_games = [g for g in games if g.round_name == round_name]
        correct = sum(1 for g in round_games if g.is_correct is True)
        total = len(round_games)
        if total > 0:
            result[round_name] = (correct, total)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_bracket_deterministic(
    year: int,
    models: list,
    feature_lookup: FeatureLookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
    feature_mode: str = "difference",
) -> SimulationResult:
    """Simulate a full bracket by always picking the higher-probability team.

    Args:
        year: Tournament year.
        models: Trained ensemble models.
        feature_lookup: For building feature vectors.
        matchups_df: Restructured matchups (for extracting R64).
        actual_bracket: Ground truth bracket for evaluation.
        feature_mode: "difference" or "raw".

    Returns:
        SimulationResult with predicted bracket and accuracy.
    """
    r64 = _extract_r64_matchups(matchups_df, year, feature_lookup)
    games, _ = _simulate_single_bracket(
        year, r64, models, feature_lookup, rng=None,
        feature_mode=feature_mode,
    )

    actual_winners = _build_actual_winners(actual_bracket)
    evaluated = _evaluate_games(games, actual_winners)

    acc = _accuracy_by_round(evaluated)
    total_correct = sum(c for c, _ in acc.values())
    total_games = sum(t for _, t in acc.values())

    bracket = Bracket(
        year=year, source="simulation", games=tuple(evaluated),
    )

    logger.info(
        "Deterministic simulation %d: %d/%d correct (%.1f%%)",
        year, total_correct, total_games,
        100.0 * total_correct / max(total_games, 1),
    )

    return SimulationResult(
        year=year,
        bracket=bracket,
        accuracy_by_round=acc,
        overall_accuracy=(total_correct, total_games),
    )


def simulate_bracket_monte_carlo(
    year: int,
    models: list,
    feature_lookup: FeatureLookup,
    matchups_df: pd.DataFrame,
    actual_bracket: Bracket,
    n_simulations: int = 1000,
    rng_seed: int = 42,
    feature_mode: str = "difference",
) -> MonteCarloResult:
    """Run N bracket simulations, sampling outcomes from model probabilities.

    Args:
        year: Tournament year.
        models: Trained ensemble models.
        feature_lookup: For building feature vectors.
        matchups_df: Restructured matchups (for extracting R64).
        actual_bracket: Ground truth bracket for evaluation.
        n_simulations: Number of simulations to run.
        rng_seed: Random seed for reproducibility.
        feature_mode: "difference" or "raw".

    Returns:
        MonteCarloResult with aggregated statistics.
    """
    r64 = _extract_r64_matchups(matchups_df, year, feature_lookup)
    actual_winners = _build_actual_winners(actual_bracket)
    rng = np.random.default_rng(rng_seed)

    # Pre-compute all game probabilities once (they don't change between sims)
    probabilities = _precompute_probabilities(
        year, r64, models, feature_lookup, feature_mode=feature_mode,
    )

    # Accumulators
    team_round_wins: dict[str, dict[str, int]] = {}
    round_correct_counts: dict[str, list[int]] = {
        r: [] for r in ROUND_ORDER
    }
    champion_counts: dict[str, int] = {}

    for _ in range(n_simulations):
        games = _simulate_from_precomputed(
            year, r64, probabilities, rng,
        )
        evaluated = _evaluate_games(games, actual_winners)

        # Track per-round accuracy
        for round_name in ROUND_ORDER:
            round_games = [
                g for g in evaluated if g.round_name == round_name
            ]
            correct = sum(1 for g in round_games if g.is_correct is True)
            round_correct_counts[round_name].append(correct)

        # Track team advancement
        for g in games:
            if g.winner is None:
                continue
            team = g.winner.team
            if team not in team_round_wins:
                team_round_wins[team] = {}
            team_round_wins[team][g.round_name] = (
                team_round_wins[team].get(g.round_name, 0) + 1
            )

        # Track champion
        ncg = [g for g in games if g.round_name == "NCG"]
        if ncg and ncg[0].winner is not None:
            champ = ncg[0].winner.team
            champion_counts[champ] = champion_counts.get(champ, 0) + 1

    # Normalize to rates
    team_round_rates: dict[str, dict[str, float]] = {}
    for team, rounds in team_round_wins.items():
        team_round_rates[team] = {
            r: count / n_simulations for r, count in rounds.items()
        }

    mean_acc: dict[str, float] = {}
    from sports_quant.march_madness._bracket import GAMES_PER_ROUND
    for round_name in ROUND_ORDER:
        counts = round_correct_counts[round_name]
        total = GAMES_PER_ROUND[round_name]
        mean_acc[round_name] = np.mean(counts) / total if total else 0.0

    champ_dist = {
        team: count / n_simulations
        for team, count in sorted(
            champion_counts.items(), key=lambda x: x[1], reverse=True,
        )
    }

    logger.info(
        "Monte Carlo simulation %d: %d sims, top champion: %s (%.1f%%)",
        year, n_simulations,
        next(iter(champ_dist), "N/A"),
        100.0 * next(iter(champ_dist.values()), 0.0),
    )

    return MonteCarloResult(
        year=year,
        n_simulations=n_simulations,
        team_round_rates=team_round_rates,
        mean_accuracy_by_round=mean_acc,
        champion_distribution=champ_dist,
    )


# ---------------------------------------------------------------------------
# Monte Carlo optimization: precompute probabilities
# ---------------------------------------------------------------------------


def _precompute_probabilities(
    year: int,
    r64_matchups: list[tuple[TeamStats, TeamStats]],
    models: list,
    feature_lookup: FeatureLookup,
    feature_mode: str = "difference",
) -> dict[tuple[str, str], float]:
    """Precompute win probability for every possible team pairing.

    Since KenPom stats are fixed pre-tournament, the probability of
    Team A beating Team B is the same regardless of which round they
    meet. We compute this once for all 64-choose-2 pairings.

    Returns:
        Dict of (team1_name, team2_name) -> P(team1 wins).
    """
    # Collect all teams from R64
    all_teams: list[TeamStats] = []
    seen: set[str] = set()
    for t1, t2 in r64_matchups:
        if t1.team not in seen:
            all_teams.append(t1)
            seen.add(t1.team)
        if t2.team not in seen:
            all_teams.append(t2)
            seen.add(t2.team)

    logger.info(
        "Precomputing probabilities for %d teams (%d pairings)",
        len(all_teams), len(all_teams) * (len(all_teams) - 1) // 2,
    )

    probs: dict[tuple[str, str], float] = {}
    for i, t1 in enumerate(all_teams):
        for t2 in all_teams[i + 1:]:
            p = _predict_game(
                t1, t2, models, feature_lookup, feature_mode,
            )
            probs[(t1.team, t2.team)] = p
            probs[(t2.team, t1.team)] = 1.0 - p

    return probs


def _simulate_from_precomputed(
    year: int,
    r64_matchups: list[tuple[TeamStats, TeamStats]],
    probabilities: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> list[BracketGame]:
    """Simulate one bracket using precomputed probabilities (fast)."""
    games: list[BracketGame] = []
    current_matchups = list(r64_matchups)

    for round_name in ROUND_ORDER:
        winners: list[TeamStats] = []

        for game_idx, (t1, t2) in enumerate(current_matchups):
            prob = probabilities[(t1.team, t2.team)]
            team1_wins = rng.random() < prob

            winner_stats = t1 if team1_wins else t2
            winners.append(winner_stats)

            slot1 = BracketSlot(team=t1.team, seed=t1.seed)
            slot2 = BracketSlot(team=t2.team, seed=t2.seed)

            game = BracketGame(
                round_name=round_name,
                region=_assign_region(round_name, game_idx),
                game_index=game_idx,
                team1=slot1,
                team2=slot2,
                winner=slot1 if team1_wins else slot2,
                win_probability=prob,
                is_upset=determine_upset(t1.seed, t2.seed, team1_wins),
                is_correct=None,
            )
            games.append(game)

        if len(winners) >= 2:
            current_matchups = [
                (winners[i], winners[i + 1])
                for i in range(0, len(winners), 2)
            ]

    return games
