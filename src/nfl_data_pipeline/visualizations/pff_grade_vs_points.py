"""Generate small multiples scatter plots of PFF grades vs total points scored."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nfl_data_pipeline import _config as config
from nfl_data_pipeline.visualizations.correlation_heatmap import _PFF_GRADE_COLS

logger = logging.getLogger(__name__)

NCOLS = 4
NROWS = 3


def _load_data() -> tuple[pd.DataFrame, int, int]:
    """Load and filter game data. Returns (df, min_season, max_season)."""
    df = pd.read_csv(config.OVERUNDER_GP)
    logger.info("Loaded %d rows from %s", len(df), config.OVERUNDER_GP)

    # Filter out pushes / no-data and week-1 zero-GP games
    df = df[df["total"] != 2].copy()
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()
    logger.info("After filtering: %d games", len(df))

    seasons = df["season"].dropna().unique()
    return df, int(min(seasons)), int(max(seasons))


def generate_pff_grade_vs_points():
    """Generate and save the PFF grade vs total points small multiples chart."""
    df, min_season, max_season = _load_data()
    n_games = len(df)

    total_points = (df["home-score"] + df["away-score"]).values

    season_label = (
        f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    )

    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"
    accent_color = "#4fc3f7"
    point_color = "#4fc3f7"

    fig, axes = plt.subplots(
        NROWS, NCOLS, figsize=(16, 11), facecolor=bg_color,
    )
    axes_flat = axes.flatten()

    for idx, (display_name, home_col, away_col) in enumerate(_PFF_GRADE_COLS):
        ax = axes_flat[idx]
        ax.set_facecolor(bg_color)

        combined_grade = (df[home_col] + df[away_col]).values / 2

        # Scatter
        ax.scatter(
            combined_grade, total_points,
            s=8, alpha=0.25, color=point_color, edgecolors="none", rasterized=True,
        )

        # OLS regression line
        mask = np.isfinite(combined_grade) & np.isfinite(total_points)
        slope, intercept, r_value, _, _ = stats.linregress(
            combined_grade[mask], total_points[mask],
        )
        x_range = np.array([combined_grade[mask].min(), combined_grade[mask].max()])
        ax.plot(x_range, intercept + slope * x_range, color="#ff7043", linewidth=1.5)

        # Pearson r annotation
        ax.text(
            0.97, 0.95, f"r = {r_value:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="white",
            bbox=dict(facecolor="#1e1e1e", edgecolor="#444444", boxstyle="round,pad=0.3"),
        )

        ax.set_title(display_name, fontsize=10, color=text_color, pad=6)
        ax.tick_params(colors=muted_color, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide the unused last cell
    for idx in range(len(_PFF_GRADE_COLS), NROWS * NCOLS):
        axes_flat[idx].set_visible(False)

    # Shared axis labels
    fig.text(0.5, 0.02, "Combined PFF Grade (avg of home + away)", ha="center",
             fontsize=10, color=muted_color)
    fig.text(0.01, 0.5, "Total Points Scored", va="center", rotation="vertical",
             fontsize=10, color=muted_color)

    # Title / subtitle
    fig.suptitle(
        "PFF Grade vs Total Points Scored",
        fontsize=15, fontweight="bold", color="white", y=0.98,
    )
    fig.text(
        0.5, 0.955,
        f"Regular season {season_label} \u00b7 {n_games:,} games",
        ha="center", fontsize=10, color=muted_color,
    )

    # Footer
    fig.text(
        0.5, 0.005,
        "Source: PFF grades + PFR/Vegas lines \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0.025, 0.04, 1, 0.94])

    config.GRADES_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.PFF_GRADE_VS_POINTS_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.PFF_GRADE_VS_POINTS_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_pff_grade_vs_points()
