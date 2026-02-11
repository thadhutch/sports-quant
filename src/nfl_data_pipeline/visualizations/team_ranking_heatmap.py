"""Generate team ranking heatmap across PFF grade categories."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)

# PFF rank columns and their display names (home-side only; we average home+away appearances)
_RANK_CATEGORIES = [
    ("off-avg", "Offense"),
    ("pass-avg", "Passing"),
    ("pblk-avg", "Pass Blk"),
    ("recv-avg", "Receiving"),
    ("run-avg", "Rushing"),
    ("rblk-avg", "Run Blk"),
    ("def-avg", "Defense"),
    ("rdef-avg", "Run Def"),
    ("tack-avg", "Tackling"),
    ("prsh-avg", "Pass Rush"),
    ("cov-avg", "Coverage"),
]


def _build_team_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-team average PFF grades across all games in the dataset,
    then rank teams 1-N within each category.

    Each team appears as both home and away throughout the season. We collect
    all of their grade observations (from home and away columns) and average.
    """
    records = []
    for _, row in df.iterrows():
        for suffix, _ in _RANK_CATEGORIES:
            home_col = f"home-{suffix}"
            away_col = f"away-{suffix}"
            # Home team's grade for this game
            records.append({
                "team": row["home_team"],
                "category": suffix,
                "grade": row[home_col],
            })
            # Away team's grade for this game
            records.append({
                "team": row["away_team"],
                "category": suffix,
                "grade": row[away_col],
            })

    grades_df = pd.DataFrame(records)
    avg_grades = grades_df.groupby(["team", "category"])["grade"].mean().unstack("category")

    # Rank within each category (highest grade = rank 1)
    rankings = avg_grades.rank(ascending=False, method="min").astype(int)

    # Rename columns to display names
    col_map = {suffix: display for suffix, display in _RANK_CATEGORIES}
    rankings = rankings.rename(columns=col_map)

    # Sort teams by average rank across all categories (best overall at top)
    rankings["avg_rank"] = rankings.mean(axis=1)
    rankings = rankings.sort_values("avg_rank")
    rankings = rankings.drop(columns="avg_rank")

    return rankings


def generate_team_ranking_heatmap(season: int | None = None):
    """Generate and save the team ranking heatmap.

    Args:
        season: Specific season to visualize. If None, uses the most recent season.
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games with no prior PFF data
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    if season is None:
        season = int(df["season"].max())
    df = df[df["season"] == season].copy()
    logger.info("Season %d: %d games", season, len(df))

    rankings = _build_team_rankings(df)
    n_teams = len(rankings)

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"

    # Custom green-to-red colormap (rank 1 = green, rank 32 = red)
    cmap = LinearSegmentedColormap.from_list(
        "rank_cmap",
        ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"],
    )

    fig_height = max(8, n_teams * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    sns.heatmap(
        rankings,
        annot=True,
        fmt="d",
        cmap=cmap,
        vmin=1,
        vmax=n_teams,
        linewidths=0.5,
        linecolor="#333333",
        annot_kws={"size": 9, "fontweight": "bold"},
        cbar_kws={"shrink": 0.6, "aspect": 30, "label": "Rank"},
        ax=ax,
    )

    # Style tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, color=text_color, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, color=text_color, rotation=0)
    ax.tick_params(axis="both", length=0)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=text_color, labelsize=8)
    cbar.set_label("Rank", color=text_color, fontsize=9)
    cbar.outline.set_edgecolor("#333333")

    # Title and subtitle
    fig.suptitle(
        f"NFL Team Rankings by PFF Grade Category",
        fontsize=15, fontweight="bold", color="white", y=0.98,
    )
    ax.set_title(
        f"{season} Regular Season \u00b7 {n_teams} teams \u00b7 Rank 1 = best",
        fontsize=10, color="#888888", pad=14,
    )

    # Footer
    fig.text(
        0.5, 0.002,
        "Source: PFF grades \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.TEAM_RANKING_HEATMAP_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.TEAM_RANKING_HEATMAP_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_team_ranking_heatmap()
