"""Generate grade stability (std dev vs games played) line chart."""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)

_CATEGORIES = [
    ("off-avg", "Offense"),
    ("def-avg", "Defense"),
    ("pass-avg", "Passing"),
    ("run-avg", "Rushing"),
    ("cov-avg", "Coverage"),
]

_COLORS = ["#4fc3f7", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6"]


def generate_grade_stability():
    """Generate and save a line chart showing grade std-dev by games played.

    At each GP level, compute the standard deviation of rolling PFF grades
    across all team-appearances. As GP grows, std-dev should shrink — showing
    when rolling averages stabilize.
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    # Collect observations tagged by games-played level
    records = []
    for _, row in df.iterrows():
        for suffix, _ in _CATEGORIES:
            records.append({
                "gp": row["home_gp"],
                "category": suffix,
                "grade": row[f"home-{suffix}"],
            })
            records.append({
                "gp": row["away_gp"],
                "category": suffix,
                "grade": row[f"away-{suffix}"],
            })

    grades_df = pd.DataFrame(records)

    # Standard deviation of grades at each GP level, per category
    stability = grades_df.groupby(["gp", "category"])["grade"].std().reset_index()
    stability.columns = ["gp", "category", "std"]

    # Filter to GP levels with enough observations (at least 10)
    counts = grades_df.groupby(["gp", "category"]).size().reset_index(name="n")
    stability = stability.merge(counts, on=["gp", "category"])
    stability = stability[stability["n"] >= 10]

    max_gp = int(stability["gp"].max())

    seasons = df["season"].unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    for (suffix, display), color in zip(_CATEGORIES, _COLORS):
        cat_data = stability[stability["category"] == suffix].sort_values("gp")
        ax.plot(cat_data["gp"], cat_data["std"], "o-", color=color,
                linewidth=1.8, markersize=4, label=display, alpha=0.85)

    ax.set_xlabel("Games Played", fontsize=10, color=text_color, labelpad=8)
    ax.set_ylabel("Std Dev of Rolling PFF Grade", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333333", labelcolor=text_color,
              loc="upper right")

    # Title and subtitle
    fig.suptitle(
        "PFF Grade Stability by Games Played",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"{season_label} \u00b7 GP 1\u2013{max_gp} \u00b7 lower std = more stable",
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
        config.GRADE_STABILITY_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.GRADE_STABILITY_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_grade_stability()
