"""Generate Vegas line accuracy histogram."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)


def _parse_spread_signed(row: pd.Series) -> float:
    """Extract a signed spread from the Vegas Line, relative to the home team.

    Positive = home team favored, negative = away team favored.
    """
    vl = row["Vegas Line"].strip()
    if vl == "Pick":
        return 0.0
    parts = vl.rsplit(" ", 1)
    spread_val = float(parts[-1])
    favorite_name = parts[0]
    if favorite_name == row["home_team"]:
        return abs(spread_val)
    return -abs(spread_val)


def generate_vegas_line_accuracy():
    """Generate and save the Vegas line accuracy histogram."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    df = df[df["Vegas Line"].notna()].copy()
    logger.info("After filtering: %d games", len(df))

    df["spread_signed"] = df.apply(_parse_spread_signed, axis=1)
    df["actual_margin"] = df["home-score"] - df["away-score"]
    df["error"] = df["actual_margin"] - df["spread_signed"]

    seasons = df["season"].dropna().unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    n_games = len(df)

    mean_err = df["error"].mean()
    std_err = df["error"].std()
    mae = df["error"].abs().mean()

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Histogram
    bins = np.arange(-50, 52, 2)
    n_vals, bin_edges, patches = ax.hist(
        df["error"], bins=bins, color="#4fc3f7", edgecolor="#0e1117",
        linewidth=0.5, alpha=0.85,
    )

    # Color bars by sign: green for home beat spread, red for home missed spread
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= 0:
            patch.set_facecolor("#2ecc71")
        else:
            patch.set_facecolor("#e74c3c")
        patch.set_alpha(0.75)

    # Mean line
    ax.axvline(x=mean_err, color="#f1c40f", linewidth=2, linestyle="-",
               label=f"Mean = {mean_err:+.1f}")
    ax.axvline(x=0, color="white", linewidth=1, linestyle="--", alpha=0.5)

    # +/- 1 std dev shading
    ax.axvspan(mean_err - std_err, mean_err + std_err, alpha=0.08, color="white",
               label=f"\u00b11 SD = {std_err:.1f}")

    # Axes styling
    ax.set_xlabel("Actual Margin \u2212 Vegas Spread (pts)", fontsize=10, color=text_color, labelpad=8)
    ax.set_ylabel("Number of Games", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(loc="upper right", fontsize=9, facecolor="#1a1a2e", edgecolor="#333333",
              labelcolor=text_color)

    # Title and subtitle
    fig.suptitle(
        "Vegas Spread Accuracy",
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
        config.VEGAS_LINE_ACCURACY_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.VEGAS_LINE_ACCURACY_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_vegas_line_accuracy()
