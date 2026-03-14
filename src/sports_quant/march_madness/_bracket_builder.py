"""Builder functions for constructing Bracket objects from data sources.

Constructs immutable Bracket instances from:
- restructured_matchups.csv (ground truth / actual results)
- backtest ensemble/debiased results CSVs (predictions)
"""

from __future__ import annotations

import logging

import pandas as pd

from sports_quant.march_madness._bracket import (
    GAMES_PER_REGION_R64,
    ROUND_NAMES,
    ROUND_ORDER,
    Bracket,
    BracketGame,
    BracketSlot,
    determine_upset,
)

logger = logging.getLogger(__name__)


def _assign_region(round_name: str, game_index: int) -> int:
    """Assign a region index (0-3) based on round and game position.

    R64 games are in groups of 8 per region (0-7=region 0, 8-15=region 1, etc.)
    Later rounds halve: R32 groups of 4, S16 groups of 2, E8 groups of 1.
    F4 and NCG are cross-region (region = -1).
    """
    if round_name in ("F4", "NCG"):
        return -1

    games_per_region = {
        "R64": 8,
        "R32": 4,
        "S16": 2,
        "E8": 1,
    }
    per_region = games_per_region.get(round_name, 1)
    return game_index // per_region


def _row_to_game(
    row: pd.Series,
    round_name: str,
    game_index: int,
    *,
    pred_col: str | None = None,
    prob_col: str | None = None,
    actual_col: str = "Team1_Win",
) -> BracketGame:
    """Convert a DataFrame row into a BracketGame.

    Args:
        row: A row from the matchups or results DataFrame.
        round_name: The round this game belongs to.
        game_index: Position of this game within its round.
        pred_col: Column name for the predicted outcome (None for actual bracket).
        prob_col: Column name for the win probability (None for actual bracket).
        actual_col: Column name for the actual outcome.
    """
    team1 = BracketSlot(team=row["Team1"], seed=int(row["Seed1"]))
    team2 = BracketSlot(team=row["Team2"], seed=int(row["Seed2"]))

    # Determine winner
    if pred_col is not None and pred_col in row.index:
        team1_wins = bool(row[pred_col] == 1)
    else:
        team1_wins = bool(row[actual_col] == 1)

    winner = team1 if team1_wins else team2

    # Win probability
    win_probability = None
    if prob_col is not None and prob_col in row.index:
        win_probability = float(row[prob_col])

    # Upset detection
    is_upset = determine_upset(team1.seed, team2.seed, team1_wins)

    # Correctness (only when we have both prediction and actual)
    is_correct = None
    if pred_col is not None and actual_col in row.index:
        actual_outcome = int(row[actual_col])
        predicted_outcome = int(row[pred_col])
        is_correct = actual_outcome == predicted_outcome

    region = _assign_region(round_name, game_index)

    return BracketGame(
        round_name=round_name,
        region=region,
        game_index=game_index,
        team1=team1,
        team2=team2,
        winner=winner,
        win_probability=win_probability,
        is_upset=is_upset,
        is_correct=is_correct,
    )


def build_actual_bracket(matchups_df: pd.DataFrame, year: int) -> Bracket:
    """Build a bracket from actual tournament results.

    Args:
        matchups_df: The restructured_matchups DataFrame (all years).
        year: The tournament year to build.

    Returns:
        A Bracket with source="actual" and all games populated.
    """
    year_df = matchups_df[matchups_df["YEAR"] == year].copy()
    logger.info("Building actual bracket for %d: %d games", year, len(year_df))

    games: list[BracketGame] = []

    for round_num in sorted(ROUND_NAMES.keys(), reverse=True):
        round_name = ROUND_NAMES[round_num]
        round_df = year_df[year_df["CURRENT ROUND"] == round_num]

        for game_idx, (_, row) in enumerate(round_df.iterrows()):
            game = _row_to_game(row, round_name, game_idx)
            games.append(game)

    return Bracket(year=year, source="actual", games=tuple(games))


def build_predicted_bracket(
    results_df: pd.DataFrame,
    matchups_df: pd.DataFrame,
    year: int,
    source: str = "ensemble",
) -> Bracket:
    """Build a bracket from backtest prediction results.

    Joins results back to matchups to recover round information.

    Args:
        results_df: Backtest results (ensemble_results.csv or debiased_results.csv).
        matchups_df: The restructured_matchups DataFrame for round info.
        year: The tournament year.
        source: Label for the prediction source ("ensemble" or "debiased").

    Returns:
        A Bracket with predictions and correctness flags.
    """
    # Determine column names based on source
    if source == "debiased":
        pred_col = "Debiased_Pred"
        prob_col = "Debiased_Prob"
    else:
        pred_col = "Ensemble_Pred"
        prob_col = "Ensemble_Prob"

    # Filter matchups to get round info for this year
    year_matchups = matchups_df[matchups_df["YEAR"] == year].reset_index(
        drop=True,
    )

    # Filter results to this year
    year_results = results_df[results_df["YEAR"] == year].copy().reset_index(
        drop=True,
    )

    # Align round info by position (results are in same row order as matchups)
    if len(year_results) != len(year_matchups):
        logger.warning(
            "Row count mismatch: %d results vs %d matchups for year %d",
            len(year_results), len(year_matchups), year,
        )

    merged = year_results.copy()
    merged["CURRENT ROUND"] = year_matchups["CURRENT ROUND"].values

    logger.info(
        "Building %s bracket for %d: %d games", source, year, len(merged),
    )

    games: list[BracketGame] = []

    for round_num in sorted(ROUND_NAMES.keys(), reverse=True):
        round_name = ROUND_NAMES[round_num]
        round_df = merged[merged["CURRENT ROUND"] == round_num]

        for game_idx, (_, row) in enumerate(round_df.iterrows()):
            game = _row_to_game(
                row, round_name, game_idx,
                pred_col=pred_col, prob_col=prob_col,
            )
            games.append(game)

    return Bracket(year=year, source=source, games=tuple(games))


def compare_brackets(
    predicted: Bracket,
    actual: Bracket,
) -> Bracket:
    """Compare a predicted bracket against the actual bracket.

    Creates a new Bracket where each game's is_correct flag reflects
    whether the predicted winner matches the actual winner.

    Args:
        predicted: The predicted bracket.
        actual: The actual bracket (ground truth).

    Returns:
        A new Bracket with is_correct flags set by comparison.
    """
    # Build lookup: (round_name, team1, team2) -> actual winner
    actual_winners: dict[tuple[str, str, str], str] = {}
    for game in actual.games:
        key = (game.round_name, game.team1.team, game.team2.team)
        actual_winners[key] = game.winner.team if game.winner else None

    compared_games: list[BracketGame] = []
    for game in predicted.games:
        key = (game.round_name, game.team1.team, game.team2.team)
        actual_winner = actual_winners.get(key)

        is_correct = None
        if actual_winner is not None and game.winner is not None:
            is_correct = game.winner.team == actual_winner

        compared_game = BracketGame(
            round_name=game.round_name,
            region=game.region,
            game_index=game.game_index,
            team1=game.team1,
            team2=game.team2,
            winner=game.winner,
            win_probability=game.win_probability,
            is_upset=game.is_upset,
            is_correct=is_correct,
        )
        compared_games.append(compared_game)

    return Bracket(
        year=predicted.year,
        source=f"{predicted.source}_vs_actual",
        games=tuple(compared_games),
    )
