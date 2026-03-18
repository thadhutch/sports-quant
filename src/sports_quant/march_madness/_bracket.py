"""Bracket data structures for March Madness tournament visualization.

Provides immutable, frozen dataclasses representing a tournament bracket
as a collection of games organized by round and region.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby


# Round number -> display name mapping
ROUND_NAMES: dict[int, str] = {
    64: "R64",
    32: "R32",
    16: "S16",
    8: "E8",
    4: "F4",
    2: "NCG",
}

# Canonical round ordering from first to last
ROUND_ORDER: tuple[str, ...] = ("R64", "R32", "S16", "E8", "F4", "NCG")

# Number of games per round in a 64-team bracket
GAMES_PER_ROUND: dict[str, int] = {
    "R64": 32,
    "R32": 16,
    "S16": 8,
    "E8": 4,
    "F4": 2,
    "NCG": 1,
}

# Number of R64 games per region
GAMES_PER_REGION_R64 = 8

# Bracket sides: regions that feed into each F4 semifinal
# F4 game 0 = region 0 winner vs region 1 winner ("left")
# F4 game 1 = region 2 winner vs region 3 winner ("right")
BRACKET_SIDES: dict[str, tuple[int, ...]] = {
    "left": (0, 1),
    "right": (2, 3),
}
SIDE_FOR_REGION: dict[int, str] = {
    0: "left", 1: "left",
    2: "right", 3: "right",
}

# Rounds that occur within regions (before F4/NCG cross-region play)
REGIONAL_ROUNDS: tuple[str, ...] = ("R64", "R32", "S16", "E8")

# Survivor pool day slots: 9 picks across the tournament.
# R64, R32, and S16 each span two days, requiring a pick per day.
# E8, F4, and NCG are single-pool slots.
SURVIVOR_SLOTS: tuple[str, ...] = (
    "R64_D1", "R64_D2",
    "R32_D1", "R32_D2",
    "S16_D1", "S16_D2",
    "E8", "F4", "NCG",
)

# Regional survivor slots (before F4/NCG cross-region play)
REGIONAL_SURVIVOR_SLOTS: tuple[str, ...] = (
    "R64_D1", "R64_D2",
    "R32_D1", "R32_D2",
    "S16_D1", "S16_D2",
    "E8",
)

# Map each survivor slot to its parent round name
SLOT_TO_ROUND: dict[str, str] = {
    "R64_D1": "R64", "R64_D2": "R64",
    "R32_D1": "R32", "R32_D2": "R32",
    "S16_D1": "S16", "S16_D2": "S16",
    "E8": "E8", "F4": "F4", "NCG": "NCG",
}


def determine_upset(seed1: int, seed2: int, team1_wins: bool) -> bool:
    """Determine if the game result is an upset.

    An upset occurs when the team with the worse (higher number) seed wins.

    Args:
        seed1: Seed of team 1.
        seed2: Seed of team 2.
        team1_wins: Whether team 1 won.

    Returns:
        True if the result is an upset.
    """
    if seed1 == seed2:
        return False
    if team1_wins:
        # Team1 won — upset if team1 has the worse (higher) seed
        return seed1 > seed2
    # Team2 won — upset if team2 has the worse (higher) seed
    return seed2 > seed1


@dataclass(frozen=True)
class BracketSlot:
    """A team occupying a position in the bracket."""

    team: str
    seed: int

    def __str__(self) -> str:
        return f"({self.seed}) {self.team}"


@dataclass(frozen=True)
class BracketGame:
    """A single game in the tournament bracket."""

    round_name: str
    region: int  # 0-3 for R64-E8; -1 for F4/NCG (cross-region)
    game_index: int  # Position within the round (0-based)
    team1: BracketSlot
    team2: BracketSlot
    winner: BracketSlot | None
    win_probability: float | None
    is_upset: bool
    is_correct: bool | None  # None = no ground truth to compare against
    game_date: str | None = None  # YYYY-MM-DD calendar date
    day_slot: str | None = None  # Survivor slot: R64_D1, R64_D2, etc.


@dataclass(frozen=True)
class Bracket:
    """A complete tournament bracket.

    Contains all games for a single year, from a single source
    (actual results, ensemble predictions, or debiased predictions).
    """

    year: int
    source: str  # "actual", "ensemble", "debiased", "ensemble_vs_actual", etc.
    games: tuple[BracketGame, ...]

    def games_by_round(self) -> dict[str, list[BracketGame]]:
        """Group games by round name, preserving ROUND_ORDER."""
        result: dict[str, list[BracketGame]] = {}
        for game in self.games:
            result.setdefault(game.round_name, []).append(game)
        return result

    def games_by_region(self) -> dict[int, list[BracketGame]]:
        """Group games by region index."""
        result: dict[int, list[BracketGame]] = {}
        for game in self.games:
            result.setdefault(game.region, []).append(game)
        return result

    def accuracy(self) -> float | None:
        """Overall prediction accuracy (only games with is_correct set).

        Returns None if no games have been evaluated.
        """
        evaluated = [g for g in self.games if g.is_correct is not None]
        if not evaluated:
            return None
        correct = sum(1 for g in evaluated if g.is_correct)
        return correct / len(evaluated)

    def accuracy_by_round(self) -> dict[str, float]:
        """Prediction accuracy broken down by round.

        Only includes rounds that have evaluated games.
        """
        by_round = self.games_by_round()
        result: dict[str, float] = {}
        for round_name in ROUND_ORDER:
            if round_name not in by_round:
                continue
            evaluated = [
                g for g in by_round[round_name] if g.is_correct is not None
            ]
            if evaluated:
                correct = sum(1 for g in evaluated if g.is_correct)
                result[round_name] = correct / len(evaluated)
        return result

    def upset_count(self) -> int:
        """Count total upsets in this bracket."""
        return sum(1 for g in self.games if g.is_upset)
