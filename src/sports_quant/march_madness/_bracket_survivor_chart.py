"""Plotly charts for survivor pool pick visualisation.

Produces interactive HTML charts showing the pick sequence, win
probabilities, and outcomes for all survivor strategies.
"""

from __future__ import annotations

import plotly.graph_objects as go

from sports_quant.march_madness._bracket_theme import DEFAULT_THEME, BracketTheme
from sports_quant.march_madness._results import SurvivorMetrics


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
        margin=dict(l=60, r=30, t=80, b=50),
    )


_STRATEGY_COLORS: dict[str, str] = {
    "greedy": "#42a5f5",
    "optimal": "#ab47bc",
    "bracket_aware": "#66bb6a",
    "mc_optimal": "#ffa726",
    "mc_optimal_seq": "#ef5350",
}


def _strategy_color(strategy: str) -> str:
    """Map strategy name to a line colour."""
    return _STRATEGY_COLORS.get(strategy, "#78909c")


def _pick_label(pick: dict) -> str:
    """X-axis label for a pick: prefer day_slot, fall back to round."""
    return pick.get("day_slot") or pick.get("round", "")


def _build_trace(
    metrics: SurvivorMetrics,
    theme: BracketTheme,
) -> go.Scatter:
    """Build a single Scatter trace for one strategy's picks."""
    x_labels = [_pick_label(p) for p in metrics.picks]
    probs = [p["win_prob"] * 100.0 for p in metrics.picks]

    marker_colors = [
        theme.survivor_survived_color if p["survived"]
        else theme.survivor_eliminated_color
        for p in metrics.picks
    ]

    labels = [
        f"[{p['seed']}] {p['team']}"
        for p in metrics.picks
    ]

    hover = [
        (
            f"<b>{p['team']}</b> ({p['seed']} seed)<br>"
            f"vs {p['opponent']}<br>"
            f"Win prob: {p['win_prob']:.1%}<br>"
            f"{'Survived' if p['survived'] else 'Eliminated'}"
        )
        for p in metrics.picks
    ]

    line_color = _strategy_color(metrics.strategy)

    return go.Scatter(
        x=x_labels,
        y=probs,
        mode="lines+markers+text",
        name=metrics.strategy.replace("_", " ").title(),
        text=labels,
        textposition="top center",
        textfont=dict(size=9, color=theme.text_dimmed),
        hovertext=hover,
        hoverinfo="text",
        line=dict(color=line_color, width=2, dash="solid"),
        marker=dict(
            size=12,
            color=marker_colors,
            line=dict(color=line_color, width=2),
            symbol="circle",
        ),
    )


def render_survivor_chart(
    survivor_metrics: list[SurvivorMetrics],
    year: int,
    *,
    theme: BracketTheme | None = None,
) -> go.Figure:
    """Create a journey chart comparing survivor strategies for one year.

    Args:
        survivor_metrics: Survivor results for this year (one per strategy).
        year: Tournament year (used in title).
        theme: Visual style (defaults to ``DEFAULT_THEME``).

    Returns:
        A Plotly ``Figure`` ready for display or export.
    """
    theme = theme or DEFAULT_THEME
    fig = go.Figure()

    for metrics in survivor_metrics:
        if not metrics.picks:
            continue
        fig.add_trace(_build_trace(metrics, theme))

    # Build subtitle with survival summary
    summaries = []
    for m in survivor_metrics:
        total = m.total_rounds
        status = "survived all" if m.survived_all else f"eliminated slot {m.rounds_survived + 1}"
        if m.exhausted:
            status = f"exhausted slot {m.rounds_survived + 1}"
        summaries.append(
            f"{m.strategy.replace('_', ' ').title()}: {m.rounds_survived}/{total} ({status})"
        )
    subtitle = " | ".join(summaries)

    fig.update_layout(
        title=dict(
            text=(
                f"Survivor Pool Picks — {year}"
                f"<br><span style='font-size:12px;color:{theme.text_dimmed}'>"
                f"{subtitle}</span>"
            ),
            font=dict(size=16),
        ),
        xaxis_title="Day Slot",
        yaxis_title="Win Probability (%)",
        yaxis_range=[0, 109],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="center",
            x=0.5,
        ),
        **_theme_layout(theme),
    )

    return fig


def render_multi_year_survivor_chart(
    all_metrics: list[SurvivorMetrics],
    *,
    theme: BracketTheme | None = None,
) -> go.Figure:
    """Create a grouped bar chart of rounds survived across years.

    Args:
        all_metrics: Survivor results across multiple years/strategies.
        theme: Visual style (defaults to ``DEFAULT_THEME``).

    Returns:
        A Plotly ``Figure`` comparing survival depth.
    """
    theme = theme or DEFAULT_THEME

    # Determine the max total_rounds across all metrics (6 or 9)
    max_total = max((m.total_rounds for m in all_metrics), default=6)

    # Group by strategy
    strategies: dict[str, list[SurvivorMetrics]] = {}
    for m in all_metrics:
        strategies.setdefault(m.strategy, []).append(m)

    fig = go.Figure()

    for strategy, metrics_list in sorted(strategies.items()):
        sorted_metrics = sorted(metrics_list, key=lambda m: m.year)
        years = [str(m.year) for m in sorted_metrics]
        rounds = [m.rounds_survived for m in sorted_metrics]
        totals = [m.total_rounds for m in sorted_metrics]
        color = _strategy_color(strategy)

        fig.add_trace(go.Bar(
            name=strategy.replace("_", " ").title(),
            x=years,
            y=rounds,
            text=[f"{r}/{t}" for r, t in zip(rounds, totals)],
            textposition="outside",
            marker_color=color,
            marker_line_color=theme.box_stroke,
            marker_line_width=1,
        ))

    fig.update_layout(
        title=dict(
            text="Survivor Pool — Slots Survived by Year",
            font=dict(size=16),
        ),
        xaxis_title="Year",
        yaxis_title="Slots Survived",
        yaxis_range=[0, max_total + 1.5],
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
