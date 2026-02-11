"""Generate a correlation heatmap of PFF grades vs scoring/O-U outcomes."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def _parse_spread(vegas_line: str) -> float:
    """Extract the numeric spread from a Vegas Line string."""
    if vegas_line.strip() == "Pick":
        return 0.0
    return abs(float(vegas_line.rsplit(" ", 1)[-1]))


# PFF grade columns: (display name, home col, away col)
_PFF_GRADE_COLS = [
    ("Offense", "home-off-avg", "away-off-avg"),
    ("Passing", "home-pass-avg", "away-pass-avg"),
    ("Pass Block", "home-pblk-avg", "away-pblk-avg"),
    ("Receiving", "home-recv-avg", "away-recv-avg"),
    ("Rushing", "home-run-avg", "away-run-avg"),
    ("Run Block", "home-rblk-avg", "away-rblk-avg"),
    ("Defense", "home-def-avg", "away-def-avg"),
    ("Run Defense", "home-rdef-avg", "away-rdef-avg"),
    ("Tackling", "home-tack-avg", "away-tack-avg"),
    ("Pass Rush", "home-prsh-avg", "away-prsh-avg"),
    ("Coverage", "home-cov-avg", "away-cov-avg"),
]

_OUTCOME_NAMES = ["O/U Line", "Total Score", "O/U Margin", "Score Diff", "Spread", "Went Over"]


def _prepare_heatmap_data() -> tuple[pd.DataFrame, int, int, int]:
    """Load data, derive features, and compute the Pearson correlation matrix.

    Returns (corr_matrix, n_games, min_season, max_season).
    """
    df = pd.read_csv(config.OVERUNDER_GP)
    logger.info("Loaded %d rows from %s", len(df), config.OVERUNDER_GP)

    # Filter out rows with no O/U result (total == 2 means push/no-data)
    df = df[df["total"] != 2].copy()

    # Filter out week-1 games where teams have 0 games played (all-zero PFF averages)
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()
    logger.info("After filtering: %d games", len(df))

    # Derive combined PFF grade features (average of home + away)
    feature_df = pd.DataFrame()
    for display_name, home_col, away_col in _PFF_GRADE_COLS:
        feature_df[display_name] = (df[home_col] + df[away_col]) / 2

    # Derive outcome/context variables
    feature_df["O/U Line"] = df["ou_line"].values
    feature_df["Total Score"] = (df["home-score"] + df["away-score"]).values
    feature_df["O/U Margin"] = feature_df["Total Score"] - feature_df["O/U Line"]
    feature_df["Score Diff"] = (df["home-score"] - df["away-score"]).values
    feature_df["Spread"] = df["Vegas Line"].apply(_parse_spread).values
    feature_df["Went Over"] = df["total"].values

    corr = feature_df.corr(method="pearson")

    seasons = df["season"].dropna().unique()
    return corr, len(feature_df), int(min(seasons)), int(max(seasons))


def generate_correlation_heatmap():
    """Generate and save the PFF grade correlation heatmap."""
    corr, n_games, min_season, max_season = _prepare_heatmap_data()

    season_label = (
        f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    )

    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    n_pff = len(_PFF_GRADE_COLS)

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        square=True,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="#333333",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8, "aspect": 30},
        ax=ax,
    )

    # Draw white separator lines between PFF grades and outcome variables
    ax.axhline(y=n_pff, color="white", linewidth=2)
    ax.axvline(x=n_pff, color="white", linewidth=2)

    # Style tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, color=text_color, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, color=text_color, rotation=0)
    ax.tick_params(axis="both", length=0)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=text_color, labelsize=8)
    cbar.outline.set_edgecolor("#333333")

    # Title and subtitle
    fig.suptitle(
        "PFF Grade Correlations with Scoring & O/U Outcomes",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax.set_title(
        f"Regular season {season_label} \u00b7 {n_games:,} games",
        fontsize=10,
        color="#888888",
        pad=14,
    )

    # Footer
    fig.text(
        0.5,
        0.005,
        "Source: PFF grades + PFR/Vegas lines \u00b7 r/sportsbetting",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    config.GRADES_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.CORRELATION_HEATMAP_CHART,
        dpi=200,
        bbox_inches="tight",
        facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.CORRELATION_HEATMAP_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_correlation_heatmap()
