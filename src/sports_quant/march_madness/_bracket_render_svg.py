"""SVG bracket renderer using ``drawsvg``.

Produces a complete SVG ``Drawing`` from a ``Bracket``, ``BracketLayout``,
and ``BracketTheme``.  All mutation is confined to the ``drawsvg.Drawing``
under construction — source data is never modified.
"""

from __future__ import annotations

import drawsvg as draw

from sports_quant.march_madness._bracket import (
    GAMES_PER_ROUND,
    ROUND_ORDER,
    Bracket,
    BracketGame,
)
from sports_quant.march_madness._bracket_layout import (
    BracketLayout,
    GamePosition,
    _COLUMN_HEADERS,
    feeder_indices,
    previous_round,
)
from sports_quant.march_madness._bracket_theme import BracketTheme


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _box_border_color(game: BracketGame, theme: BracketTheme) -> str:
    """Choose border colour based on correctness / upset status."""
    if game.is_correct is True:
        return theme.correct_color
    if game.is_correct is False:
        return theme.incorrect_color
    if game.is_upset:
        return theme.upset_color
    return theme.box_stroke


def _confidence_opacity(game: BracketGame, theme: BracketTheme) -> float:
    """Map win_probability to fill opacity."""
    if game.win_probability is None:
        return theme.confidence_max_opacity
    t = max(0.0, min(1.0, game.win_probability))
    return theme.confidence_min_opacity + t * (
        theme.confidence_max_opacity - theme.confidence_min_opacity
    )


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _add_shadow_filter(d: draw.Drawing, theme: BracketTheme) -> str:
    """Register a drop-shadow SVG filter and return its url reference."""
    filt = draw.Raw(
        f'<filter id="shadow" x="-5%" y="-5%" width="115%" height="115%">'
        f'<feDropShadow dx="{theme.shadow_offset}" dy="{theme.shadow_offset}" '
        f'stdDeviation="{theme.shadow_blur}" flood-color="{theme.shadow_color}" />'
        f"</filter>"
    )
    d.append(filt)
    return "url(#shadow)"


def _draw_game_box(
    d: draw.Drawing,
    game: BracketGame,
    pos: GamePosition,
    theme: BracketTheme,
    shadow_url: str,
) -> None:
    """Draw a single game box with seed badges and team names."""
    x, y = pos.x, pos.y
    w, h = theme.box_width, theme.box_height
    opacity = _confidence_opacity(game, theme)
    border = _box_border_color(game, theme)

    # Outer box (with shadow)
    d.append(draw.Rectangle(
        x, y, w, h,
        rx=theme.box_corner_radius,
        ry=theme.box_corner_radius,
        fill=theme.box_fill,
        fill_opacity=opacity,
        stroke=border,
        stroke_width=theme.box_stroke_width,
        filter=shadow_url,
    ))

    # Separator line between slots
    sep_y = y + theme.slot_height
    d.append(draw.Line(
        x + 1, sep_y, x + w - 1, sep_y,
        stroke=theme.separator_color,
        stroke_width=theme.separator_width,
    ))

    # Draw each slot (team1 = top, team2 = bottom)
    for slot_idx, slot in enumerate([game.team1, game.team2]):
        slot_y = y + slot_idx * theme.slot_height
        is_winner = game.winner is not None and slot == game.winner

        # Seed badge background
        d.append(draw.Rectangle(
            x, slot_y, theme.seed_badge_width, theme.slot_height,
            rx=theme.box_corner_radius if slot_idx == 0 else 0,
            ry=theme.box_corner_radius if slot_idx == 0 else 0,
            fill=theme.seed_badge_color,
        ))
        # Round bottom-left corner for bottom slot
        if slot_idx == 1:
            d.append(draw.Rectangle(
                x, slot_y, theme.seed_badge_width, theme.slot_height,
                rx=theme.box_corner_radius,
                ry=theme.box_corner_radius,
                fill=theme.seed_badge_color,
                clip_path=f"inset(0 0 0 0)",
            ))

        # Seed number
        seed_x = x + theme.seed_badge_width / 2.0
        seed_y = slot_y + theme.slot_height / 2.0
        d.append(draw.Text(
            str(slot.seed),
            theme.seed_font_size,
            seed_x,
            seed_y,
            fill=theme.seed_color,
            font_family=theme.font_family,
            font_weight="bold",
            text_anchor="middle",
            dominant_baseline="central",
        ))

        # Team name
        name_x = x + theme.seed_badge_width + 4.0
        name_y = slot_y + theme.slot_height / 2.0
        name_color = theme.text_color if is_winner else theme.text_dimmed
        name_weight = "bold" if is_winner else "normal"

        # Truncate long names
        display_name = slot.team
        max_chars = 14
        if len(display_name) > max_chars:
            display_name = display_name[: max_chars - 1] + "…"

        d.append(draw.Text(
            display_name,
            theme.team_font_size,
            name_x,
            name_y,
            fill=name_color,
            font_family=theme.font_family,
            font_weight=name_weight,
            text_anchor="start",
            dominant_baseline="central",
        ))

    # Win probability annotation (top-right corner)
    if game.win_probability is not None:
        prob_text = f"{game.win_probability:.0%}"
        prob_x = x + w - 4.0
        prob_y = y + theme.slot_height / 2.0
        d.append(draw.Text(
            prob_text,
            theme.prob_font_size,
            prob_x,
            prob_y,
            fill=theme.text_dimmed,
            font_family=theme.font_family,
            text_anchor="end",
            dominant_baseline="central",
        ))


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------

def _draw_connector(
    d: draw.Drawing,
    feeder_pos: GamePosition,
    game_pos: GamePosition,
    feeder_side: str,
    theme: BracketTheme,
) -> None:
    """Draw a single connector line from a feeder game to the current game.

    For left-side feeders the output is on the right edge; for right-side
    feeders the output is on the left edge.
    """
    bw = theme.box_width
    bh = theme.box_height

    if feeder_side in ("left", "center"):
        start_x = feeder_pos.x + bw
        end_x = game_pos.x
    else:
        start_x = feeder_pos.x
        end_x = game_pos.x + bw

    start_y = feeder_pos.y + bh / 2.0
    end_y = game_pos.y + bh / 2.0
    mid_x = (start_x + end_x) / 2.0

    p = draw.Path(
        stroke=theme.connector_color,
        stroke_width=theme.connector_width,
        fill="none",
    )
    p.M(start_x, start_y)
    p.C(mid_x, start_y, mid_x, end_y, end_x, end_y)
    d.append(p)


def _draw_all_connectors(
    d: draw.Drawing,
    layout: BracketLayout,
    theme: BracketTheme,
    game_keys: set[tuple[str, int]],
) -> None:
    """Draw bracket connectors for every non-R64 game present in the bracket."""
    for round_name in ROUND_ORDER[1:]:
        prev = previous_round(round_name)
        if prev is None:
            continue
        n_games = GAMES_PER_ROUND[round_name]
        for gi in range(n_games):
            key = (round_name, gi)
            if key not in game_keys:
                continue
            feeders = feeder_indices(round_name, gi)
            if feeders is None:
                continue
            game_pos = layout.game_positions[key]
            for fi in feeders:
                feeder_key = (prev, fi)
                if feeder_key not in game_keys:
                    continue
                feeder_pos = layout.game_positions[feeder_key]
                _draw_connector(d, feeder_pos, game_pos, feeder_pos.side, theme)


# ---------------------------------------------------------------------------
# Headers, title, legend
# ---------------------------------------------------------------------------

def _draw_title(d: draw.Drawing, bracket: Bracket, theme: BracketTheme) -> None:
    """Draw the bracket title centred at the top."""
    title = f"{bracket.year} NCAA Tournament — {bracket.source.replace('_', ' ').title()}"
    cx = float(d.width) / 2.0  # type: ignore[arg-type]
    ty = theme.canvas_padding + theme.title_height * 0.6
    d.append(draw.Text(
        title,
        theme.title_font_size,
        cx,
        ty,
        fill=theme.title_color,
        font_family=theme.font_family,
        font_weight="bold",
        text_anchor="middle",
        dominant_baseline="central",
    ))


def _draw_headers(
    d: draw.Drawing,
    layout: BracketLayout,
    theme: BracketTheme,
) -> None:
    """Draw round-name headers above each column."""
    for col, (hx, hy) in layout.column_header_positions.items():
        label = _COLUMN_HEADERS.get(col, "")
        d.append(draw.Text(
            label,
            theme.header_font_size,
            hx,
            hy,
            fill=theme.header_color,
            font_family=theme.font_family,
            font_weight="bold",
            text_anchor="middle",
            dominant_baseline="central",
        ))


def _draw_legend(
    d: draw.Drawing,
    theme: BracketTheme,
    has_correctness: bool,
) -> None:
    """Draw a small colour legend at the bottom-left."""
    entries: list[tuple[str, str]] = []
    if has_correctness:
        entries.append((theme.correct_color, "Correct"))
        entries.append((theme.incorrect_color, "Incorrect"))
    entries.append((theme.upset_color, "Upset"))

    lx = theme.canvas_padding
    ly = float(d.height) - theme.canvas_padding * 0.6  # type: ignore[arg-type]
    s = theme.legend_swatch_size

    for i, (colour, label) in enumerate(entries):
        offset = i * (s + theme.legend_gap + 40)
        d.append(draw.Rectangle(
            lx + offset, ly - s / 2, s, s,
            rx=2, ry=2,
            fill=colour,
        ))
        d.append(draw.Text(
            label,
            theme.legend_font_size,
            lx + offset + s + 4,
            ly,
            fill=theme.text_dimmed,
            font_family=theme.font_family,
            text_anchor="start",
            dominant_baseline="central",
        ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_bracket_svg(
    bracket: Bracket,
    layout: BracketLayout,
    theme: BracketTheme,
) -> draw.Drawing:
    """Render a complete bracket as an SVG ``Drawing``.

    Args:
        bracket: The bracket data to visualise.
        layout: Pre-computed game positions.
        theme: Visual style constants.

    Returns:
        A ``drawsvg.Drawing`` ready to be saved or exported.
    """
    d = draw.Drawing(layout.width, layout.height)

    # Background
    d.append(draw.Rectangle(
        0, 0, layout.width, layout.height,
        fill=theme.background_color,
    ))

    shadow_url = _add_shadow_filter(d, theme)

    # Build lookup of which games exist
    game_lookup: dict[tuple[str, int], BracketGame] = {
        (g.round_name, g.game_index): g for g in bracket.games
    }
    game_keys = set(game_lookup.keys())

    # Connectors first (behind boxes)
    _draw_all_connectors(d, layout, theme, game_keys)

    # Game boxes
    for key, game in game_lookup.items():
        if key in layout.game_positions:
            pos = layout.game_positions[key]
            _draw_game_box(d, game, pos, theme, shadow_url)

    # Chrome
    _draw_title(d, bracket, theme)
    _draw_headers(d, layout, theme)

    has_correctness = any(g.is_correct is not None for g in bracket.games)
    _draw_legend(d, theme, has_correctness)

    return d
