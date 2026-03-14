"""Design system constants for bracket visualization.

All visual properties — colors, dimensions, typography, spacing — live here
in a single frozen dataclass so renderers stay declarative.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BracketTheme:
    """Immutable theme controlling all visual properties of a bracket."""

    # Canvas
    background_color: str = "#1a1a2e"
    canvas_padding: float = 50.0

    # Title & headers
    title_height: float = 45.0
    header_height: float = 30.0
    header_color: str = "#90caf9"
    title_color: str = "#e0e0e0"
    title_font_size: float = 18.0
    header_font_size: float = 12.0

    # Game box
    box_width: float = 140.0
    box_height: float = 44.0
    slot_height: float = 22.0
    box_corner_radius: float = 4.0
    box_fill: str = "#16213e"
    box_stroke: str = "#2a3a5c"
    box_stroke_width: float = 1.0

    # Seed badge (left section of each slot)
    seed_badge_width: float = 26.0
    seed_badge_color: str = "#0d1b2a"

    # Outcome colors
    correct_color: str = "#00c853"
    incorrect_color: str = "#ff1744"
    upset_color: str = "#ffd700"
    neutral_fill: str = "#16213e"

    # Text colors
    text_color: str = "#ffffff"
    text_dimmed: str = "#78909c"
    seed_color: str = "#b0bec5"

    # Typography
    font_family: str = "Arial, Helvetica, sans-serif"
    team_font_size: float = 10.5
    seed_font_size: float = 10.0
    prob_font_size: float = 9.0

    # Layout spacing
    column_gap: float = 24.0
    r64_gap: float = 8.0
    region_gap: float = 30.0
    num_columns: int = 11  # cols 0-10

    # Connectors
    connector_color: str = "#4a5568"
    connector_width: float = 1.5

    # Confidence visualization
    confidence_min_opacity: float = 0.40
    confidence_max_opacity: float = 1.0

    # Shadow (SVG filter)
    shadow_blur: float = 2.0
    shadow_offset: float = 2.0
    shadow_color: str = "rgba(0,0,0,0.3)"

    # Separator line inside game box
    separator_color: str = "#2a3a5c"
    separator_width: float = 0.5

    # Legend
    legend_font_size: float = 9.0
    legend_swatch_size: float = 10.0
    legend_gap: float = 16.0


DEFAULT_THEME = BracketTheme()
