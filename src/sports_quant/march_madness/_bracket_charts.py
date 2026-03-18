"""Plotly-based analytical charts for bracket performance.

Produces interactive HTML and static PNG charts for accuracy-by-round
and multi-bracket comparison views.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from sports_quant.march_madness._bracket import ROUND_ORDER, Bracket
from sports_quant.march_madness._bracket_theme import DEFAULT_THEME, BracketTheme


def _theme_layout(theme: BracketTheme) -> dict:
    """Return a Plotly layout dict styled to match the bracket theme."""
    return dict(
        font=dict(family=theme.font_family, color=theme.text_color),
        plot_bgcolor=theme.background_color,
        paper_bgcolor=theme.background_color,
        xaxis=dict(
            gridcolor=theme.box_stroke,
            zerolinecolor=theme.box_stroke,
        ),
        yaxis=dict(
            gridcolor=theme.box_stroke,
            zerolinecolor=theme.box_stroke,
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )


def render_accuracy_chart(
    bracket: Bracket,
    *,
    theme: BracketTheme | None = None,
) -> go.Figure:
    """Create a bar chart of prediction accuracy by round.

    Args:
        bracket: A bracket with ``is_correct`` flags set.
        theme: Visual style (defaults to ``DEFAULT_THEME``).

    Returns:
        A Plotly ``Figure`` ready for display or export.
    """
    theme = theme or DEFAULT_THEME
    acc = bracket.accuracy_by_round()

    rounds = [r for r in ROUND_ORDER if r in acc]
    values = [acc[r] * 100.0 for r in rounds]
    labels = [f"{v:.0f}%" for v in values]

    colors = [
        theme.correct_color if v >= 60 else theme.incorrect_color
        for v in values
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=rounds,
                y=values,
                text=labels,
                textposition="outside",
                marker_color=colors,
                marker_line_color=theme.box_stroke,
                marker_line_width=1,
            ),
        ],
    )

    fig.update_layout(
        title=dict(
            text=f"Accuracy by Round — {bracket.year} ({bracket.source})",
            font=dict(size=16),
        ),
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 109],
        showlegend=False,
        **_theme_layout(theme),
    )

    return fig


def render_multi_accuracy_chart(
    brackets: list[Bracket],
    *,
    theme: BracketTheme | None = None,
) -> go.Figure:
    """Compare accuracy across multiple brackets (years or sources).

    Args:
        brackets: Two or more brackets to compare.
        theme: Visual style (defaults to ``DEFAULT_THEME``).

    Returns:
        A grouped bar chart ``Figure``.
    """
    theme = theme or DEFAULT_THEME

    palette = [
        theme.correct_color,
        theme.upset_color,
        "#42a5f5",
        "#ab47bc",
        "#ef5350",
        "#26a69a",
    ]

    fig = go.Figure()

    for idx, bracket in enumerate(brackets):
        acc = bracket.accuracy_by_round()
        rounds = [r for r in ROUND_ORDER if r in acc]
        values = [acc[r] * 100.0 for r in rounds]
        color = palette[idx % len(palette)]
        label = f"{bracket.year} {bracket.source}"

        fig.add_trace(go.Bar(
            name=label,
            x=rounds,
            y=values,
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            marker_color=color,
            marker_line_color=theme.box_stroke,
            marker_line_width=1,
        ))

    fig.update_layout(
        title=dict(
            text="Accuracy by Round — Comparison",
            font=dict(size=16),
        ),
        xaxis_title="Round",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 115],
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        **_theme_layout(theme),
    )

    return fig


def save_chart(
    fig: go.Figure,
    output_path: Path | str,
) -> Path:
    """Save a Plotly figure to disk (HTML or static image).

    The format is inferred from the file extension:
    - ``.html`` → interactive HTML
    - ``.png`` / ``.svg`` / ``.pdf`` → static image (requires kaleido or orca)

    Args:
        fig: The Plotly figure.
        output_path: Destination file path.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    if ext == ".html":
        fig.write_html(str(output_path), include_plotlyjs="cdn")
    else:
        fig.write_image(str(output_path))

    return output_path
