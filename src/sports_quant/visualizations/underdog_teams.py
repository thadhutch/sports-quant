"""Generate team-level underdog SU win rate charts with NFL logos."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox

from sports_quant import _config as config
from sports_quant.visualizations.logos import get_logo_image

logger = logging.getLogger(__name__)


def _parse_spread(vegas_line: str) -> float:
    """Extract the numeric spread from a Vegas Line string."""
    if vegas_line.strip() == "Pick":
        return 0.0
    return abs(float(vegas_line.rsplit(" ", 1)[-1]))


# Map historical franchise names to their current name so stats consolidate
_FRANCHISE_RENAMES: dict[str, str] = {
    "Oakland Raiders": "Las Vegas Raiders",
    "San Diego Chargers": "Los Angeles Chargers",
    "St. Louis Rams": "Los Angeles Rams",
    "Washington Redskins": "Washington Commanders",
    "Washington Football Team": "Washington Commanders",
}


def _load_underdog_data() -> pd.DataFrame:
    """Load game data and compute underdog / winner columns."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Consolidate relocated/renamed franchises
    df["home_team"] = df["home_team"].replace(_FRANCHISE_RENAMES)
    df["away_team"] = df["away_team"].replace(_FRANCHISE_RENAMES)

    df["spread"] = df["Vegas Line"].apply(_parse_spread)
    df["favorite_team"] = df["Vegas Line"].apply(
        lambda vl: vl.rsplit(" ", 1)[0] if vl.strip() != "Pick" else None
    )

    # Determine underdog team
    df["underdog"] = df.apply(
        lambda r: (
            r["away_team"]
            if r["favorite_team"] == r["home_team"]
            else r["home_team"]
        )
        if r["favorite_team"] is not None
        else None,
        axis=1,
    )

    # Determine winner
    df["winner"] = df.apply(
        lambda r: (
            r["home_team"]
            if r["home-score"] > r["away-score"]
            else (r["away_team"] if r["away-score"] > r["home-score"] else None)
        ),
        axis=1,
    )

    # Exclude ties and pick'ems
    df = df[df["winner"].notna() & (df["spread"] > 0) & df["underdog"].notna()].copy()
    df["upset"] = df["winner"] == df["underdog"]
    return df


def _build_team_stats(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Group by underdog team, compute win % and filter by minimum sample size."""
    stats = (
        df.groupby("underdog")
        .agg(wins=("upset", "sum"), games=("upset", "count"))
        .reset_index()
    )
    stats["win_pct"] = (stats["wins"] / stats["games"] * 100).round(1)
    stats = stats[stats["games"] >= min_games].sort_values("win_pct", ascending=True)
    return stats


def _render_chart(
    stats: pd.DataFrame,
    title: str,
    subtitle: str,
    output_path,
):
    """Render a horizontal bar chart with team logos on the y-axis."""
    bg_color = "#0e1117"
    text_color = "#e0e0e0"

    n = len(stats)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Color gradient: worst (bottom, red) → best (top, green)
    cmap = plt.cm.RdYlGn
    norm_vals = np.linspace(0, 1, n)
    colors = [cmap(v) for v in norm_vals]

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, stats["win_pct"].values, color=colors, height=0.7)

    # Percentage + sample size labels at end of bars
    for i, (bar, (_, row)) in enumerate(zip(bars, stats.iterrows())):
        label = f"{row['win_pct']:.1f}%"
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        ax.annotate(
            f"(n={int(row['games'])})",
            xy=(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset fontsize",
            ha="left",
            va="center",
            fontsize=7,
            color="#888888",
        )

    # Team logos on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * n)  # blank text labels; logos replace them

    for i, (_, row) in enumerate(stats.iterrows()):
        try:
            logo = get_logo_image(row["underdog"])
            ab = AnnotationBbox(
                logo,
                (-0.5, i),
                xycoords=("axes fraction", "data"),
                box_alignment=(1.0, 0.5),
                frameon=False,
            )
            ax.add_artist(ab)
        except KeyError:
            # Fallback: show team name as text if logo unavailable
            ax.text(
                -1,
                i,
                row["underdog"],
                ha="right",
                va="center",
                fontsize=8,
                color=text_color,
                transform=ax.get_yaxis_transform(),
            )

    # Axis styling
    max_pct = stats["win_pct"].max()
    ax.set_xlim(0, min(max_pct + 20, 100))
    ax.set_xlabel("Straight-Up Win %", fontsize=10, color=text_color, labelpad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#333333", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=text_color, length=0)
    ax.tick_params(axis="y", length=0)

    # Title / subtitle
    fig.suptitle(title, fontsize=14, fontweight="bold", color="white", y=0.98)
    ax.set_title(subtitle, fontsize=10, color="#888888", pad=14)

    # Footer
    fig.text(
        0.5,
        0.005,
        "Source: PFR/Vegas lines \u00b7 r/sportsbetting",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    plt.tight_layout(rect=[0.08, 0.02, 1, 0.95])

    config.LINE_ANALYSIS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=bg_color)
    logger.info("Chart saved to %s", output_path)
    plt.close(fig)


def generate_best_underdog_teams_chart():
    """Generate the best underdog teams (all spreads) chart."""
    df = _load_underdog_data()
    stats = _build_team_stats(df)

    seasons = df["season"].dropna().unique()
    min_s, max_s = int(min(seasons)), int(max(seasons))
    season_label = f"{min_s}\u2013{max_s}" if min_s != max_s else str(min_s)

    _render_chart(
        stats,
        title="Best Underdog Teams (SU Win Rate)",
        subtitle=f"Regular season {season_label} \u00b7 {int(stats['games'].sum()):,} games as underdog",
        output_path=config.BEST_UNDERDOG_TEAMS_CHART,
    )


def generate_dogs_that_bite_chart():
    """Generate the dogs that bite (7+ point underdogs) chart."""
    df = _load_underdog_data()
    df = df[df["spread"] >= 7].copy()
    stats = _build_team_stats(df)

    seasons = df["season"].dropna().unique()
    min_s, max_s = int(min(seasons)), int(max(seasons))
    season_label = f"{min_s}\u2013{max_s}" if min_s != max_s else str(min_s)

    _render_chart(
        stats,
        title="Dogs That Bite: Best Teams at 7+ Point Underdogs",
        subtitle=f"Regular season {season_label} \u00b7 {int(stats['games'].sum()):,} games as 7+ pt underdog",
        output_path=config.DOGS_THAT_BITE_CHART,
    )


if __name__ == "__main__":
    generate_best_underdog_teams_chart()
    generate_dogs_that_bite_chart()
