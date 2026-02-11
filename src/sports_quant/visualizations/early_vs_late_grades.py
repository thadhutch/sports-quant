"""Generate early-season vs late-season PFF grade comparison chart."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)

_CORE_CATEGORIES = [
    ("off-avg", "Offense"),
    ("def-avg", "Defense"),
    ("pass-avg", "Passing"),
    ("run-avg", "Rushing"),
    ("cov-avg", "Coverage"),
]


def generate_early_vs_late_grades():
    """Generate and save a grouped bar chart comparing early vs late season grades.

    Early = games 1-4 (gp 1-4), Late = games 13+ (gp >= 13).
    Uses home_gp / away_gp as proxy for game number.
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games with no prior PFF data
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    # Collect observations tagged with game-number bucket
    records = []
    for _, row in df.iterrows():
        # Home team observation
        home_gp = row["home_gp"]
        if home_gp <= 4:
            bucket = "early"
        elif home_gp >= 13:
            bucket = "late"
        else:
            bucket = None

        if bucket:
            for suffix, _ in _CORE_CATEGORIES:
                records.append({
                    "bucket": bucket,
                    "category": suffix,
                    "grade": row[f"home-{suffix}"],
                })

        # Away team observation
        away_gp = row["away_gp"]
        if away_gp <= 4:
            bucket = "early"
        elif away_gp >= 13:
            bucket = "late"
        else:
            bucket = None

        if bucket:
            for suffix, _ in _CORE_CATEGORIES:
                records.append({
                    "bucket": bucket,
                    "category": suffix,
                    "grade": row[f"away-{suffix}"],
                })

    grades_df = pd.DataFrame(records)
    pivot = grades_df.groupby(["bucket", "category"])["grade"].mean().unstack("bucket")

    # Order categories to match _CORE_CATEGORIES
    suffixes = [s for s, _ in _CORE_CATEGORIES]
    labels = [d for _, d in _CORE_CATEGORIES]
    early_vals = [pivot.loc[s, "early"] for s in suffixes]
    late_vals = [pivot.loc[s, "late"] for s in suffixes]

    seasons = df["season"].unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)

    n_early = len(grades_df[grades_df["bucket"] == "early"]) // len(_CORE_CATEGORIES)
    n_late = len(grades_df[grades_df["bucket"] == "late"]) // len(_CORE_CATEGORIES)

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    bars_early = ax.bar(x - width / 2, early_vals, width, label="Early (GP 1\u20134)",
                        color="#4fc3f7", alpha=0.85, edgecolor="#333333", linewidth=0.5)
    bars_late = ax.bar(x + width / 2, late_vals, width, label="Late (GP 13+)",
                       color="#f1c40f", alpha=0.85, edgecolor="#333333", linewidth=0.5)

    # Value labels on bars
    for bars in [bars_early, bars_late]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=8, color="#aaaaaa")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=text_color)
    ax.set_ylabel("Average PFF Grade", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(axis="y", colors=muted_color, labelsize=9)
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333333", labelcolor=text_color)

    # Title and subtitle
    fig.suptitle(
        "Early Season vs Late Season PFF Grades",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"{season_label} \u00b7 {n_early} early obs \u00b7 {n_late} late obs",
        fontsize=10, color=muted_color, pad=14,
    )

    # Footer
    fig.text(
        0.5, 0.005,
        "Source: PFF grades \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    config.GRADES_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.EARLY_VS_LATE_GRADES_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.EARLY_VS_LATE_GRADES_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_early_vs_late_grades()
