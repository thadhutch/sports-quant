"""Public API for bracket visualisation.

Orchestrates layout computation, SVG rendering, and file export.
All functions return the output path(s) they wrote to.

Usage::

    from sports_quant.march_madness.bracket_plots import render_bracket
    render_bracket(bracket, "output/bracket.svg")
"""

from __future__ import annotations

from pathlib import Path

from sports_quant.march_madness._bracket import Bracket
from sports_quant.march_madness._bracket_charts import (
    render_accuracy_chart as _accuracy_chart,
    render_multi_accuracy_chart as _multi_accuracy_chart,
    save_chart,
)
from sports_quant.march_madness._bracket_layout import compute_layout
from sports_quant.march_madness._bracket_render_svg import render_bracket_svg
from sports_quant.march_madness._bracket_theme import DEFAULT_THEME, BracketTheme


def _has_cairosvg() -> bool:
    """Check whether CairoSVG (and its system Cairo lib) are available."""
    try:
        import cairosvg  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _save_drawing(drawing, output_path: Path) -> Path:
    """Save a ``drawsvg.Drawing`` based on file extension.

    SVG always works.  PNG and PDF require the optional ``cairosvg``
    package and the system Cairo library (``brew install cairo`` on macOS).
    Falls back to SVG with a warning when Cairo is unavailable.
    """
    import logging

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()

    if ext in (".png", ".pdf") and not _has_cairosvg():
        logging.getLogger(__name__).warning(
            "cairosvg not available — falling back to SVG. "
            "Install cairosvg and the system Cairo library for %s export.",
            ext.upper(),
        )
        output_path = output_path.with_suffix(".svg")
        ext = ".svg"

    if ext == ".png":
        drawing.save_png(str(output_path))
    elif ext == ".pdf":
        import cairosvg

        svg_bytes = drawing.as_svg().encode("utf-8")
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(output_path))
    else:
        drawing.save_svg(str(output_path))

    return output_path


def render_bracket(
    bracket: Bracket,
    output_path: str | Path,
    *,
    theme: BracketTheme | None = None,
) -> Path:
    """Render a bracket visualisation to SVG, PNG, or PDF.

    The format is inferred from the file extension (``.svg``, ``.png``,
    ``.pdf``).  Defaults to SVG.

    Args:
        bracket: The bracket data to visualise.
        output_path: Destination file path.
        theme: Visual style (defaults to ``DEFAULT_THEME``).

    Returns:
        The resolved output path.
    """
    theme = theme or DEFAULT_THEME
    layout = compute_layout(theme)
    drawing = render_bracket_svg(bracket, layout, theme)
    return _save_drawing(drawing, Path(output_path))


def render_comparison(
    predicted: Bracket,
    actual: Bracket,
    output_path: str | Path,
    *,
    theme: BracketTheme | None = None,
) -> Path:
    """Render a comparison bracket (predicted vs actual).

    Uses ``compare_brackets`` to flag correctness, then renders the
    resulting bracket with green/red colour coding.

    Args:
        predicted: The predicted bracket.
        actual: The actual bracket (ground truth).
        output_path: Destination file path.
        theme: Visual style (defaults to ``DEFAULT_THEME``).

    Returns:
        The resolved output path.
    """
    from sports_quant.march_madness._bracket_builder import compare_brackets

    compared = compare_brackets(predicted, actual)
    return render_bracket(compared, output_path, theme=theme)


def render_accuracy(
    bracket: Bracket,
    output_path: str | Path,
    *,
    theme: BracketTheme | None = None,
) -> Path:
    """Render an accuracy-by-round chart.

    Supports ``.html`` (interactive) and ``.png`` / ``.svg`` (static).

    Args:
        bracket: A bracket with ``is_correct`` flags set.
        output_path: Destination file path.
        theme: Visual style.

    Returns:
        The resolved output path.
    """
    fig = _accuracy_chart(bracket, theme=theme)
    return save_chart(fig, output_path)


def render_accuracy_comparison(
    brackets: list[Bracket],
    output_path: str | Path,
    *,
    theme: BracketTheme | None = None,
) -> Path:
    """Render a multi-bracket accuracy comparison chart.

    Args:
        brackets: Two or more brackets to compare.
        output_path: Destination file path.
        theme: Visual style.

    Returns:
        The resolved output path.
    """
    fig = _multi_accuracy_chart(brackets, theme=theme)
    return save_chart(fig, output_path)
