"""Generate O/U line vs actual total score scatter plot."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def generate_ou_line_vs_actual():
    """Generate and save the O/U line vs actual total score chart."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out pushes/no-data and rows with missing O/U line
    df = df[df["total"] != 2].copy()
    df = df[df["ou_line"].notna()].copy()
    logger.info("After filtering: %d games", len(df))

    df["actual_total"] = df["home-score"] + df["away-score"]
    df["went_over"] = df["actual_total"] > df["ou_line"]
    df["went_under"] = df["actual_total"] < df["ou_line"]

    seasons = df["season"].dropna().unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    n_games = len(df)

    over_pct = df["went_over"].mean() * 100
    under_pct = df["went_under"].mean() * 100

    # OLS regression
    mask = np.isfinite(df["ou_line"]) & np.isfinite(df["actual_total"])
    slope, intercept, r_value, _, _ = stats.linregress(
        df.loc[mask, "ou_line"], df.loc[mask, "actual_total"]
    )

    # MAE (mean absolute error of the line)
    mae = (df["actual_total"] - df["ou_line"]).abs().mean()

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Plot overs and unders separately
    overs = df[df["went_over"]]
    unders = df[df["went_under"]]
    pushes = df[~df["went_over"] & ~df["went_under"]]

    ax.scatter(
        overs["ou_line"], overs["actual_total"],
        s=12, alpha=0.35, color="#2ecc71", edgecolors="none", rasterized=True,
        label=f"Over ({over_pct:.1f}%)",
    )
    ax.scatter(
        unders["ou_line"], unders["actual_total"],
        s=12, alpha=0.35, color="#e74c3c", edgecolors="none", rasterized=True,
        label=f"Under ({under_pct:.1f}%)",
    )
    if len(pushes) > 0:
        ax.scatter(
            pushes["ou_line"], pushes["actual_total"],
            s=12, alpha=0.5, color="#f1c40f", edgecolors="none", rasterized=True,
            label="Push",
        )

    # Perfect-line diagonal (where O/U line exactly matches actual score)
    line_range = [df["ou_line"].min() - 2, df["ou_line"].max() + 2]
    ax.plot(line_range, line_range, color="white", linewidth=1.2, linestyle="--",
            alpha=0.6, label="Perfect line")

    # Regression line
    x_range = np.array(line_range)
    ax.plot(x_range, intercept + slope * x_range, color="#4fc3f7", linewidth=1.5,
            label=f"OLS (r={r_value:.2f})")

    # Axes styling
    ax.set_xlabel("Vegas O/U Line", fontsize=10, color=text_color, labelpad=8)
    ax.set_ylabel("Actual Total Score", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e", edgecolor="#333333",
              labelcolor=text_color)

    # Title and subtitle
    fig.suptitle(
        "O/U Line vs Actual Total Score",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"Regular season {season_label} \u00b7 {n_games:,} games \u00b7 MAE = {mae:.1f} pts",
        fontsize=10, color=muted_color, pad=14,
    )

    # Footer
    fig.text(
        0.5, 0.005,
        "Source: PFR/Vegas lines \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    config.LINE_ANALYSIS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.OU_LINE_VS_ACTUAL_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.OU_LINE_VS_ACTUAL_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_ou_line_vs_actual()
