"""Pure coordinate math for bracket layout.

Computes (x, y) positions for all 63 games in a standard 64-team bracket
without referencing any rendering library.  Every function is deterministic
and operates on immutable data.

Column assignments (11 columns, 0-10):
    0-3   Left rounds   (R64, R32, S16, E8)
    4     F4 left semifinal
    5     NCG (championship)
    6     F4 right semifinal
    7-10  Right rounds  (E8, S16, R32, R64)
"""

from __future__ import annotations

from dataclasses import dataclass

from sports_quant.march_madness._bracket import (
    GAMES_PER_ROUND,
    ROUND_ORDER,
)
from sports_quant.march_madness._bracket_theme import BracketTheme


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

_LEFT_COLUMNS: dict[str, int] = {"R64": 0, "R32": 1, "S16": 2, "E8": 3}
_RIGHT_COLUMNS: dict[str, int] = {"R64": 10, "R32": 9, "S16": 8, "E8": 7}
_F4_LEFT_COL = 4
_NCG_COL = 5
_F4_RIGHT_COL = 6

# Number of games per region in each round
_GAMES_PER_REGION: dict[str, int] = {"R64": 8, "R32": 4, "S16": 2, "E8": 1}

# Round headers for each column
_COLUMN_HEADERS: dict[int, str] = {
    0: "R64",
    1: "R32",
    2: "Sweet 16",
    3: "Elite 8",
    4: "Final 4",
    5: "Championship",
    6: "Final 4",
    7: "Elite 8",
    8: "Sweet 16",
    9: "R32",
    10: "R64",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GamePosition:
    """Screen coordinates for one game box."""

    x: float
    y: float
    side: str  # "left", "right", or "center"


@dataclass(frozen=True)
class BracketLayout:
    """Pre-computed positions for every game in a 64-team bracket."""

    width: float
    height: float
    game_positions: dict[tuple[str, int], GamePosition]
    column_header_positions: dict[int, tuple[float, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _region_for(round_name: str, game_index: int) -> int:
    """Return region (0-3) for rounds R64-E8, or -1 for F4/NCG."""
    if round_name in ("F4", "NCG"):
        return -1
    per_region = _GAMES_PER_REGION[round_name]
    return game_index // per_region


def _side_for(round_name: str, game_index: int) -> str:
    """Return 'left', 'right', or 'center' for a game."""
    if round_name == "NCG":
        return "center"
    if round_name == "F4":
        return "left" if game_index == 0 else "right"
    region = _region_for(round_name, game_index)
    return "left" if region <= 1 else "right"


def _column_for(round_name: str, game_index: int) -> int:
    """Map a game to its column index (0-10)."""
    if round_name == "NCG":
        return _NCG_COL
    if round_name == "F4":
        return _F4_LEFT_COL if game_index == 0 else _F4_RIGHT_COL
    region = _region_for(round_name, game_index)
    if region <= 1:
        return _LEFT_COLUMNS[round_name]
    return _RIGHT_COLUMNS[round_name]


def _column_x(col: int, theme: BracketTheme) -> float:
    """Convert column index to pixel x coordinate."""
    return theme.canvas_padding + col * (theme.box_width + theme.column_gap)


# ---------------------------------------------------------------------------
# Y-position computation
# ---------------------------------------------------------------------------

def _compute_y_positions(theme: BracketTheme) -> dict[tuple[str, int], float]:
    """Compute y-centre of every game box using recursive midpoint logic.

    R64 games are evenly spaced.  Every later round is centred between its
    two feeder games from the previous round.
    """
    stride = theme.box_height + theme.r64_gap
    top_offset = theme.canvas_padding + theme.title_height + theme.header_height

    y: dict[tuple[str, int], float] = {}

    # R64: 32 games, 8 per region, 4 regions
    for gi in range(GAMES_PER_ROUND["R64"]):
        region = gi // 8
        local = gi % 8
        # Regions 0,1 are left side; 2,3 are right side.
        # Top regions (0,2) start at top_offset; bottom regions (1,3) follow.
        is_bottom_region = region in (1, 3)
        region_start = top_offset
        if is_bottom_region:
            region_start += 8 * stride + theme.region_gap
        y[("R64", gi)] = region_start + local * stride

    # R32 through NCG: each game centred between its two feeders
    for round_name in ROUND_ORDER[1:]:  # skip R64
        prev_round = ROUND_ORDER[ROUND_ORDER.index(round_name) - 1]
        n_games = GAMES_PER_ROUND[round_name]
        for gi in range(n_games):
            feeder_a = 2 * gi
            feeder_b = 2 * gi + 1
            y[(round_name, gi)] = (
                y[(prev_round, feeder_a)] + y[(prev_round, feeder_b)]
            ) / 2.0

    return y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_layout(theme: BracketTheme) -> BracketLayout:
    """Compute positions for all 63 games in a standard 64-team bracket.

    Returns a ``BracketLayout`` with pre-computed pixel coordinates.  The
    layout is independent of bracket *data* — it only depends on the theme's
    dimensional constants.
    """
    y_positions = _compute_y_positions(theme)

    positions: dict[tuple[str, int], GamePosition] = {}
    for round_name in ROUND_ORDER:
        n_games = GAMES_PER_ROUND[round_name]
        for gi in range(n_games):
            col = _column_for(round_name, gi)
            x = _column_x(col, theme)
            y = y_positions[(round_name, gi)]
            side = _side_for(round_name, gi)
            positions[(round_name, gi)] = GamePosition(x=x, y=y, side=side)

    # Canvas dimensions
    last_col_x = _column_x(theme.num_columns - 1, theme) + theme.box_width
    width = last_col_x + theme.canvas_padding

    max_y = max(p.y for p in positions.values())
    height = max_y + theme.box_height + theme.canvas_padding

    # Round column headers
    header_y = theme.canvas_padding + theme.title_height
    headers: dict[int, tuple[float, float]] = {}
    for col, label in _COLUMN_HEADERS.items():
        hx = _column_x(col, theme) + theme.box_width / 2.0
        headers[col] = (hx, header_y)

    return BracketLayout(
        width=width,
        height=height,
        game_positions=positions,
        column_header_positions=headers,
    )


def feeder_indices(round_name: str, game_index: int) -> tuple[int, int] | None:
    """Return the two feeder game_indices from the previous round.

    Returns ``None`` for R64 (no feeders).  For every other round the
    feeders in the previous round are at indices ``2 * game_index`` and
    ``2 * game_index + 1``.
    """
    if round_name == "R64":
        return None
    return (2 * game_index, 2 * game_index + 1)


def previous_round(round_name: str) -> str | None:
    """Return the round that feeds into *round_name*, or ``None`` for R64."""
    idx = ROUND_ORDER.index(round_name)
    if idx == 0:
        return None
    return ROUND_ORDER[idx - 1]
