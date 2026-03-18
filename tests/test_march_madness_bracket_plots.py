"""Tests for Phase 2 bracket visualization — layout math, SVG rendering, charts."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sports_quant.march_madness._bracket import (
    GAMES_PER_ROUND,
    ROUND_ORDER,
    Bracket,
    BracketGame,
    BracketSlot,
)
from sports_quant.march_madness._bracket_builder import _assign_region
from sports_quant.march_madness._bracket_layout import (
    BracketLayout,
    GamePosition,
    _column_for,
    _side_for,
    compute_layout,
    feeder_indices,
    previous_round,
)
from sports_quant.march_madness._bracket_theme import DEFAULT_THEME, BracketTheme


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_full_bracket(
    year: int = 2024,
    source: str = "test",
) -> Bracket:
    """Build a complete 63-game bracket with deterministic data."""
    games: list[BracketGame] = []

    seed_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    for round_name in ROUND_ORDER:
        n = GAMES_PER_ROUND[round_name]
        for gi in range(n):
            region = _assign_region(round_name, gi)
            local = gi % max(1, n // 4) if round_name not in ("F4", "NCG") else gi
            seed1 = seed_pairs[local % len(seed_pairs)][0]
            seed2 = seed_pairs[local % len(seed_pairs)][1]

            team1 = BracketSlot(f"{round_name}_{gi}_A", seed1)
            team2 = BracketSlot(f"{round_name}_{gi}_B", seed2)

            games.append(BracketGame(
                round_name=round_name,
                region=region,
                game_index=gi,
                team1=team1,
                team2=team2,
                winner=team1,
                win_probability=0.50 + (gi % 5) * 0.10,
                is_upset=gi % 7 == 0,
                is_correct=True if gi % 3 != 0 else False,
            ))

    return Bracket(year=year, source=source, games=tuple(games))


def _make_partial_bracket() -> Bracket:
    """A bracket with fewer than 63 games (R64 + R32 only)."""
    games: list[BracketGame] = []
    for round_name in ("R64", "R32"):
        for gi in range(GAMES_PER_ROUND[round_name]):
            region = _assign_region(round_name, gi)
            games.append(BracketGame(
                round_name=round_name,
                region=region,
                game_index=gi,
                team1=BracketSlot(f"T{gi}A", 1),
                team2=BracketSlot(f"T{gi}B", 16),
                winner=BracketSlot(f"T{gi}A", 1),
                win_probability=0.7,
                is_upset=False,
                is_correct=None,
            ))
    return Bracket(year=2024, source="partial", games=tuple(games))


# ===========================================================================
# Theme tests
# ===========================================================================

class TestBracketTheme:
    def test_immutability(self):
        with pytest.raises(AttributeError):
            DEFAULT_THEME.background_color = "#000000"

    def test_default_values(self):
        assert DEFAULT_THEME.box_width == 140.0
        assert DEFAULT_THEME.num_columns == 11
        assert DEFAULT_THEME.background_color == "#1a1a2e"

    def test_custom_theme(self):
        theme = BracketTheme(box_width=200.0)
        assert theme.box_width == 200.0
        assert theme.box_height == 44.0  # other defaults unchanged


# ===========================================================================
# Layout tests
# ===========================================================================

class TestLayoutHelpers:
    def test_previous_round_r64_is_none(self):
        assert previous_round("R64") is None

    def test_previous_round_r32_is_r64(self):
        assert previous_round("R32") == "R64"

    def test_previous_round_ncg_is_f4(self):
        assert previous_round("NCG") == "F4"

    def test_feeder_indices_r64_returns_none(self):
        assert feeder_indices("R64", 0) is None

    def test_feeder_indices_r32(self):
        assert feeder_indices("R32", 0) == (0, 1)
        assert feeder_indices("R32", 3) == (6, 7)

    def test_feeder_indices_ncg(self):
        assert feeder_indices("NCG", 0) == (0, 1)

    def test_side_left_regions(self):
        assert _side_for("R64", 0) == "left"   # region 0
        assert _side_for("R64", 15) == "left"  # region 1

    def test_side_right_regions(self):
        assert _side_for("R64", 16) == "right"  # region 2
        assert _side_for("R64", 31) == "right"  # region 3

    def test_side_f4(self):
        assert _side_for("F4", 0) == "left"
        assert _side_for("F4", 1) == "right"

    def test_side_ncg_is_center(self):
        assert _side_for("NCG", 0) == "center"

    def test_column_left_rounds(self):
        assert _column_for("R64", 0) == 0
        assert _column_for("R32", 0) == 1
        assert _column_for("S16", 0) == 2
        assert _column_for("E8", 0) == 3

    def test_column_right_rounds(self):
        assert _column_for("R64", 24) == 10
        assert _column_for("R32", 12) == 9
        assert _column_for("S16", 4) == 8
        assert _column_for("E8", 2) == 7

    def test_column_center(self):
        assert _column_for("F4", 0) == 4
        assert _column_for("F4", 1) == 6
        assert _column_for("NCG", 0) == 5


class TestComputeLayout:
    def test_returns_bracket_layout(self):
        layout = compute_layout(DEFAULT_THEME)
        assert isinstance(layout, BracketLayout)

    def test_has_63_positions(self):
        layout = compute_layout(DEFAULT_THEME)
        assert len(layout.game_positions) == 63

    def test_all_rounds_present(self):
        layout = compute_layout(DEFAULT_THEME)
        rounds_present = {key[0] for key in layout.game_positions}
        assert rounds_present == set(ROUND_ORDER)

    def test_positive_dimensions(self):
        layout = compute_layout(DEFAULT_THEME)
        assert layout.width > 0
        assert layout.height > 0

    def test_left_r64_x_less_than_right_r64_x(self):
        layout = compute_layout(DEFAULT_THEME)
        left_r64 = layout.game_positions[("R64", 0)]
        right_r64 = layout.game_positions[("R64", 24)]
        assert left_r64.x < right_r64.x

    def test_r32_y_is_midpoint_of_r64_feeders(self):
        layout = compute_layout(DEFAULT_THEME)
        r32_pos = layout.game_positions[("R32", 0)]
        r64_a = layout.game_positions[("R64", 0)]
        r64_b = layout.game_positions[("R64", 1)]
        expected_y = (r64_a.y + r64_b.y) / 2.0
        assert r32_pos.y == pytest.approx(expected_y)

    def test_s16_y_is_midpoint_of_r32_feeders(self):
        layout = compute_layout(DEFAULT_THEME)
        s16_pos = layout.game_positions[("S16", 0)]
        r32_a = layout.game_positions[("R32", 0)]
        r32_b = layout.game_positions[("R32", 1)]
        expected_y = (r32_a.y + r32_b.y) / 2.0
        assert s16_pos.y == pytest.approx(expected_y)

    def test_ncg_y_is_midpoint_of_f4_feeders(self):
        layout = compute_layout(DEFAULT_THEME)
        ncg = layout.game_positions[("NCG", 0)]
        f4_a = layout.game_positions[("F4", 0)]
        f4_b = layout.game_positions[("F4", 1)]
        expected_y = (f4_a.y + f4_b.y) / 2.0
        assert ncg.y == pytest.approx(expected_y)

    def test_region_gap_separates_top_and_bottom(self):
        """Bottom region R64 games should start below top region + gap."""
        layout = compute_layout(DEFAULT_THEME)
        top_last = layout.game_positions[("R64", 7)]   # last in region 0
        bottom_first = layout.game_positions[("R64", 8)]  # first in region 1
        gap = bottom_first.y - top_last.y
        expected_min = DEFAULT_THEME.box_height + DEFAULT_THEME.region_gap
        assert gap >= expected_min

    def test_column_headers_present(self):
        layout = compute_layout(DEFAULT_THEME)
        assert len(layout.column_header_positions) == 11

    def test_left_right_symmetry(self):
        """Left-side and right-side R64 should have matching y positions."""
        layout = compute_layout(DEFAULT_THEME)
        for i in range(16):
            left = layout.game_positions[("R64", i)]
            right = layout.game_positions[("R64", i + 16)]
            assert left.y == pytest.approx(right.y)


# ===========================================================================
# SVG rendering tests
# ===========================================================================

class TestRenderBracketSvg:
    def test_produces_drawing(self):
        from sports_quant.march_madness._bracket_render_svg import render_bracket_svg

        bracket = _make_full_bracket()
        layout = compute_layout(DEFAULT_THEME)
        drawing = render_bracket_svg(bracket, layout, DEFAULT_THEME)
        assert drawing is not None

    def test_svg_is_valid_xml(self):
        from sports_quant.march_madness._bracket_render_svg import render_bracket_svg

        bracket = _make_full_bracket()
        layout = compute_layout(DEFAULT_THEME)
        drawing = render_bracket_svg(bracket, layout, DEFAULT_THEME)
        svg_str = drawing.as_svg()
        # Should parse without error
        ET.fromstring(svg_str)

    def test_svg_contains_team_names(self):
        from sports_quant.march_madness._bracket_render_svg import render_bracket_svg

        bracket = _make_full_bracket()
        layout = compute_layout(DEFAULT_THEME)
        drawing = render_bracket_svg(bracket, layout, DEFAULT_THEME)
        svg_str = drawing.as_svg()
        # Check a few team names from the fixture
        assert "R64_0_A" in svg_str
        assert "NCG_0_A" in svg_str

    def test_svg_contains_title(self):
        from sports_quant.march_madness._bracket_render_svg import render_bracket_svg

        bracket = _make_full_bracket()
        layout = compute_layout(DEFAULT_THEME)
        drawing = render_bracket_svg(bracket, layout, DEFAULT_THEME)
        svg_str = drawing.as_svg()
        assert "2024" in svg_str

    def test_partial_bracket_renders(self):
        """A bracket with < 63 games should render without error."""
        from sports_quant.march_madness._bracket_render_svg import render_bracket_svg

        bracket = _make_partial_bracket()
        layout = compute_layout(DEFAULT_THEME)
        drawing = render_bracket_svg(bracket, layout, DEFAULT_THEME)
        svg_str = drawing.as_svg()
        ET.fromstring(svg_str)

    def test_comparison_bracket_has_green_and_red(self):
        """A bracket with is_correct flags should use correct/incorrect colours."""
        from sports_quant.march_madness._bracket_render_svg import render_bracket_svg

        bracket = _make_full_bracket()
        layout = compute_layout(DEFAULT_THEME)
        drawing = render_bracket_svg(bracket, layout, DEFAULT_THEME)
        svg_str = drawing.as_svg()
        assert DEFAULT_THEME.correct_color in svg_str
        assert DEFAULT_THEME.incorrect_color in svg_str


# ===========================================================================
# File export tests (smoke tests)
# ===========================================================================

class TestRenderBracketFile:
    def test_render_svg_creates_file(self, tmp_path: Path):
        from sports_quant.march_madness.bracket_plots import render_bracket

        bracket = _make_full_bracket()
        out = render_bracket(bracket, tmp_path / "bracket.svg")
        assert out.exists()
        assert out.suffix == ".svg"
        assert out.stat().st_size > 0

    def test_render_svg_valid_xml(self, tmp_path: Path):
        from sports_quant.march_madness.bracket_plots import render_bracket

        bracket = _make_full_bracket()
        out = render_bracket(bracket, tmp_path / "bracket.svg")
        ET.parse(str(out))  # raises on invalid XML

    def test_render_png_creates_file(self, tmp_path: Path):
        """PNG export requires system Cairo library (brew install cairo)."""
        from sports_quant.march_madness.bracket_plots import _has_cairosvg, render_bracket

        bracket = _make_full_bracket()
        out = render_bracket(bracket, tmp_path / "bracket.png")
        assert out.exists()

        if _has_cairosvg():
            assert out.suffix == ".png"
            assert out.stat().st_size > 1000
        else:
            # Graceful fallback to SVG
            assert out.suffix == ".svg"
            assert out.stat().st_size > 0

    def test_render_comparison_creates_file(self, tmp_path: Path):
        from sports_quant.march_madness.bracket_plots import render_comparison

        actual = _make_full_bracket(source="actual")
        predicted = _make_full_bracket(source="ensemble")
        out = render_comparison(predicted, actual, tmp_path / "comparison.svg")
        assert out.exists()


# ===========================================================================
# Chart tests
# ===========================================================================

class TestBracketCharts:
    def test_accuracy_chart_returns_figure(self):
        from sports_quant.march_madness._bracket_charts import render_accuracy_chart

        bracket = _make_full_bracket()
        fig = render_accuracy_chart(bracket)
        assert fig is not None
        assert len(fig.data) == 1

    def test_accuracy_chart_has_correct_rounds(self):
        from sports_quant.march_madness._bracket_charts import render_accuracy_chart

        bracket = _make_full_bracket()
        fig = render_accuracy_chart(bracket)
        x_values = list(fig.data[0].x)
        assert "R64" in x_values
        assert "NCG" in x_values

    def test_multi_accuracy_chart(self):
        from sports_quant.march_madness._bracket_charts import (
            render_multi_accuracy_chart,
        )

        b1 = _make_full_bracket(year=2023, source="ensemble")
        b2 = _make_full_bracket(year=2024, source="ensemble")
        fig = render_multi_accuracy_chart([b1, b2])
        assert len(fig.data) == 2

    def test_save_chart_html(self, tmp_path: Path):
        from sports_quant.march_madness._bracket_charts import (
            render_accuracy_chart,
            save_chart,
        )

        bracket = _make_full_bracket()
        fig = render_accuracy_chart(bracket)
        out = save_chart(fig, tmp_path / "accuracy.html")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_render_accuracy_api(self, tmp_path: Path):
        from sports_quant.march_madness.bracket_plots import render_accuracy

        bracket = _make_full_bracket()
        out = render_accuracy(bracket, tmp_path / "accuracy.html")
        assert out.exists()
