"""Generate Vegas line accuracy box plots broken down by surface and roof type."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)

# Group the granular surface types into two buckets
_SURFACE_MAP = {
    "grass": "Grass",
    "fieldturf": "Turf",
    "sportturf": "Turf",
    "matrixturf": "Turf",
    "astroturf": "Turf",
    "a_turf": "Turf",
    "astroplay": "Turf",
}

# Group retractable variants together
_ROOF_MAP = {
    "outdoors": "Outdoors",
    "dome": "Dome",
    "retractable roof (closed)": "Retractable",
    "retractable roof (open)": "Retractable",
}


def _parse_spread_signed(row: pd.Series) -> float:
    """Extract a signed spread relative to the home team."""
    vl = row["Vegas Line"].strip()
    if vl == "Pick":
        return 0.0
    parts = vl.rsplit(" ", 1)
    spread_val = float(parts[-1])
    favorite_name = parts[0]
    if favorite_name == row["home_team"]:
        return abs(spread_val)
    return -abs(spread_val)


def generate_vegas_accuracy_by_conditions():
    """Generate and save the Vegas accuracy by surface/roof box plots."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    df = df[df["Vegas Line"].notna()].copy()
    df["spread_signed"] = df.apply(_parse_spread_signed, axis=1)
    df["actual_margin"] = df["home-score"] - df["away-score"]
    df["error"] = df["actual_margin"] - df["spread_signed"]

    df["surface_group"] = df["Surface"].map(_SURFACE_MAP)
    df["roof_group"] = df["Roof"].map(_ROOF_MAP)
    df = df[df["surface_group"].notna() & df["roof_group"].notna()].copy()
    logger.info("After filtering: %d games", len(df))

    # Build condition label: "Outdoors / Grass", etc.
    df["condition"] = df["roof_group"] + " / " + df["surface_group"]

    # Order by median error (ascending)
    condition_order = (
        df.groupby("condition")["error"]
        .median()
        .sort_values()
        .index.tolist()
    )

    seasons = df["season"].dropna().unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    n_games = len(df)

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Prepare data in order for box plot
    box_data = [df.loc[df["condition"] == c, "error"].values for c in condition_order]
    counts = [len(d) for d in box_data]

    bp = ax.boxplot(
        box_data,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, markerfacecolor="#555555",
                        markeredgecolor="#555555", alpha=0.4),
    )

    # Style the boxes
    box_colors = ["#2ecc71", "#4fc3f7", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6"]
    for i, (patch, median) in enumerate(zip(bp["boxes"], bp["medians"])):
        color = box_colors[i % len(box_colors)]
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(text_color)
        median.set_color("white")
        median.set_linewidth(2)
    for whisker in bp["whiskers"]:
        whisker.set_color(muted_color)
        whisker.set_linewidth(0.8)
    for cap in bp["caps"]:
        cap.set_color(muted_color)
        cap.set_linewidth(0.8)

    # Zero reference line
    ax.axhline(y=0, color="white", linewidth=1, linestyle="--", alpha=0.4)

    # X-axis labels with sample sizes
    labels = [f"{c}\n(n={n:,})" for c, n in zip(condition_order, counts)]
    ax.set_xticks(range(1, len(condition_order) + 1))
    ax.set_xticklabels(labels, fontsize=9, color=text_color)

    # Axes styling
    ax.set_ylabel("Actual Margin \u2212 Vegas Spread (pts)", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(axis="y", colors=muted_color, labelsize=9)
    ax.tick_params(axis="x", colors=text_color, length=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Title and subtitle
    fig.suptitle(
        "Vegas Spread Accuracy by Playing Conditions",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"Regular season {season_label} \u00b7 {n_games:,} games",
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
        config.VEGAS_ACCURACY_CONDITIONS_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.VEGAS_ACCURACY_CONDITIONS_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_vegas_accuracy_by_conditions()
