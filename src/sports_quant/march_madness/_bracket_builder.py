"""Builder functions for constructing Bracket objects from data sources.

Constructs immutable Bracket instances from:
- restructured_matchups.csv (ground truth / actual results)
- backtest ensemble/debiased results CSVs (predictions)
"""

from __future__ import annotations

import logging

import pandas as pd

from sports_quant.march_madness._bracket import (
    GAMES_PER_ROUND,
    GAMES_PER_REGION_R64,
    ROUND_NAMES,
    ROUND_ORDER,
    Bracket,
    BracketGame,
    BracketSlot,
    determine_upset,
)
from sports_quant.march_madness._features import standardize_team_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bracket-tree ordering
# ---------------------------------------------------------------------------

# Canonical seed-pair order within each region of 8 R64 games.
# This is the standard NCAA bracket layout.
_CANONICAL_SEED_PAIR_POS: dict[tuple[int, int], int] = {
    (1, 16): 0, (8, 9): 1, (5, 12): 2, (4, 13): 3,
    (6, 11): 4, (3, 14): 5, (7, 10): 6, (2, 15): 7,
}


def _seed_pair_key(row: pd.Series) -> tuple[int, int]:
    """Return the normalised (low, high) seed pair for a game row."""
    s1, s2 = int(row["Seed1"]), int(row["Seed2"])
    return (min(s1, s2), max(s1, s2))


def _canonicalize_order(
    ordered: dict[str, list[int]],
    year_df: pd.DataFrame,
) -> dict[str, list[int]]:
    """Fix within-region ordering to match the canonical bracket layout.

    The tree walk groups games into correct regions but the within-pair
    ordering depends on Team1/Team2 CSV ordering.  This function:

    1. Sorts R64 games within each region by canonical seed-pair position.
    2. Re-derives every later round's ordering from the fixed R64 order
       by mapping each game to the position of its feeder games.
    """

    def _clean(name: str) -> str:
        return standardize_team_name(str(name).strip())

    def _winner_name(df_idx: int) -> str:
        row = year_df.loc[df_idx]
        raw = row["Team1"] if row["Team1_Win"] == 1 else row["Team2"]
        return _clean(raw)

    # Step 1 — Sort R64 within each region (groups of 8)
    r64 = list(ordered["R64"])
    sorted_r64: list[int] = []
    for start in range(0, 32, 8):
        region = r64[start : start + 8]
        region.sort(
            key=lambda idx: _CANONICAL_SEED_PAIR_POS.get(
                _seed_pair_key(year_df.loc[idx]), 99,
            ),
        )
        sorted_r64.extend(region)

    result: dict[str, list[int]] = {"R64": sorted_r64}

    # Step 2 — Re-derive R32 through NCG from the sorted R64
    prev_round_name = "R64"
    for round_name in ROUND_ORDER[1:]:
        prev_indices = result[prev_round_name]

        # Map: winner team name → sorted position in previous round
        winner_pos: dict[str, int] = {}
        for pos, idx in enumerate(prev_indices):
            winner_pos[_winner_name(idx)] = pos

        # Each game in this round maps to position = min(feeder_pos) // 2
        games_with_target: list[tuple[int, int]] = []
        for idx in ordered[round_name]:
            row = year_df.loc[idx]
            pos1 = winner_pos.get(_clean(row["Team1"]), 999)
            pos2 = winner_pos.get(_clean(row["Team2"]), 999)
            target = min(pos1, pos2) // 2
            games_with_target.append((target, idx))

        games_with_target.sort(key=lambda x: x[0])
        result[round_name] = [idx for _, idx in games_with_target]

        prev_round_name = round_name

    return result


def _bracket_tree_order(
    year_df: pd.DataFrame,
) -> dict[str, list[int]] | None:
    """Compute correct bracket-tree ordering from tournament results.

    Walks from the championship game (NCG) backward to R64, matching
    each game's participants to the winners of games in the previous round.
    Then canonicalises the within-region ordering to match the standard
    NCAA bracket layout (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15).

    Args:
        year_df: Matchups DataFrame filtered to a single year.
                 Must contain ``CURRENT ROUND``, ``Team1``, ``Team2``,
                 and ``Team1_Win`` columns.

    Returns:
        A dict of ``round_name -> [DataFrame indices in bracket order]``,
        or ``None`` if the data is incomplete and tree reconstruction
        cannot proceed (callers should fall back to raw row order).
    """

    def _clean(name: str) -> str:
        return standardize_team_name(str(name).strip())

    def _winner(row: pd.Series) -> str:
        raw = row["Team1"] if row["Team1_Win"] == 1 else row["Team2"]
        return _clean(raw)

    # Split into per-round DataFrames
    round_dfs: dict[str, pd.DataFrame] = {}
    for round_num in sorted(ROUND_NAMES.keys(), reverse=True):
        round_name = ROUND_NAMES[round_num]
        rdf = year_df[year_df["CURRENT ROUND"] == round_num]
        if len(rdf) != GAMES_PER_ROUND[round_name]:
            logger.debug(
                "Incomplete round %s (%d/%d games) — skipping tree order",
                round_name, len(rdf), GAMES_PER_ROUND[round_name],
            )
            return None
        round_dfs[round_name] = rdf

    # --- Phase 1: tree walk (NCG → R64) for region grouping ---

    ordered: dict[str, list[int]] = {
        "NCG": list(round_dfs["NCG"].index),
    }

    for round_idx in range(len(ROUND_ORDER) - 1, 0, -1):
        parent_round = ROUND_ORDER[round_idx]
        child_round = ROUND_ORDER[round_idx - 1]

        child_df = round_dfs[child_round]
        winner_to_idx: dict[str, int] = {}
        for idx, row in child_df.iterrows():
            winner_to_idx[_winner(row)] = idx

        child_ordered: list[int] = []
        for parent_idx in ordered[parent_round]:
            parent_row = year_df.loc[parent_idx]

            child1_idx = winner_to_idx.get(_clean(parent_row["Team1"]))
            child2_idx = winner_to_idx.get(_clean(parent_row["Team2"]))

            if child1_idx is None or child2_idx is None:
                logger.debug(
                    "Cannot trace tree: %s participants [%s, %s] "
                    "not found as %s winners",
                    parent_round, parent_row["Team1"],
                    parent_row["Team2"], child_round,
                )
                return None

            child_ordered.append(child1_idx)
            child_ordered.append(child2_idx)

        ordered[child_round] = child_ordered

    # --- Phase 2: canonical seed-pair ordering within each region ---

    return _canonicalize_order(ordered, year_df)


def _build_games_ordered(
    df: pd.DataFrame,
    ordered: dict[str, list[int]],
    *,
    pred_col: str | None = None,
    prob_col: str | None = None,
) -> list[BracketGame]:
    """Build ``BracketGame`` list using pre-computed bracket-tree order."""
    games: list[BracketGame] = []
    for round_name in ROUND_ORDER:
        for game_idx, df_idx in enumerate(ordered[round_name]):
            row = df.loc[df_idx]
            game = _row_to_game(
                row, round_name, game_idx,
                pred_col=pred_col, prob_col=prob_col,
            )
            games.append(game)
    return games


def _build_games_fallback(
    df: pd.DataFrame,
    *,
    pred_col: str | None = None,
    prob_col: str | None = None,
) -> list[BracketGame]:
    """Build ``BracketGame`` list using raw CSV row order (legacy path)."""
    games: list[BracketGame] = []
    for round_num in sorted(ROUND_NAMES.keys(), reverse=True):
        round_name = ROUND_NAMES[round_num]
        round_df = df[df["CURRENT ROUND"] == round_num]
        for game_idx, (_, row) in enumerate(round_df.iterrows()):
            game = _row_to_game(
                row, round_name, game_idx,
                pred_col=pred_col, prob_col=prob_col,
            )
            games.append(game)
    return games


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
    team1 = BracketSlot(
        team=standardize_team_name(str(row["Team1"]).strip()),
        seed=int(row["Seed1"]),
    )
    team2 = BracketSlot(
        team=standardize_team_name(str(row["Team2"]).strip()),
        seed=int(row["Seed2"]),
    )

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

    Uses bracket-tree reconstruction to assign correct ``game_index``
    values regardless of CSV row ordering.  Falls back to raw row
    order when the data is incomplete (< 63 games).

    Args:
        matchups_df: The restructured_matchups DataFrame (all years).
        year: The tournament year to build.

    Returns:
        A Bracket with source="actual" and all games populated.
    """
    year_df = matchups_df[matchups_df["YEAR"] == year].copy()
    logger.info("Building actual bracket for %d: %d games", year, len(year_df))

    ordered = _bracket_tree_order(year_df)

    if ordered is not None:
        games = _build_games_ordered(year_df, ordered)
    else:
        games = _build_games_fallback(year_df)

    return Bracket(year=year, source="actual", games=tuple(games))


def build_predicted_bracket(
    results_df: pd.DataFrame,
    matchups_df: pd.DataFrame,
    year: int,
    source: str = "ensemble",
) -> Bracket:
    """Build a bracket from backtest prediction results.

    Joins results back to matchups to recover round information, then
    uses bracket-tree reconstruction (from the *actual* outcomes in
    matchups) to assign correct ``game_index`` values.

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

    # Use actual outcomes (from matchups) to determine bracket tree order;
    # apply the same ordering to the merged predictions DataFrame.
    ordered = _bracket_tree_order(year_matchups)

    if ordered is not None:
        games = _build_games_ordered(
            merged, ordered, pred_col=pred_col, prob_col=prob_col,
        )
    else:
        games = _build_games_fallback(
            merged, pred_col=pred_col, prob_col=prob_col,
        )

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
