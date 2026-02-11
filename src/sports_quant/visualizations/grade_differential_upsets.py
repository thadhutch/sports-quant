"""Generate biggest grade-differential upsets horizontal bar chart."""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def generate_grade_differential_upsets(top_n: int = 20):
    """Generate and save a horizontal bar chart of the biggest PFF grade-gap upsets.

    An "upset" is a game where the lower-graded team (by composite PFF) won.

    Args:
        top_n: Number of top upsets to display.
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games with no prior PFF data
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    # Composite grade per side: average of offense + defense
    df["home_composite"] = (df["home-off-avg"] + df["home-def-avg"]) / 2
    df["away_composite"] = (df["away-off-avg"] + df["away-def-avg"]) / 2

    # Grade differential: positive means home was higher-graded
    df["grade_diff"] = df["home_composite"] - df["away_composite"]

    # Determine winner
    df["home_won"] = df["home-score"] > df["away-score"]

    # An upset: higher-graded team lost
    # If grade_diff > 0 (home favored) but away won, or vice versa
    df["upset"] = (
        ((df["grade_diff"] > 0) & (~df["home_won"])) |
        ((df["grade_diff"] < 0) & (df["home_won"]))
    )

    upsets = df[df["upset"]].copy()
    upsets["abs_diff"] = upsets["grade_diff"].abs()
    upsets = upsets.nlargest(top_n, "abs_diff")
    logger.info("Found %d upsets, showing top %d", df["upset"].sum(), top_n)

    # Build labels and determine winner/loser for display
    labels = []
    for _, row in upsets.iterrows():
        if row["home_won"]:
            winner, loser = row["home_team"], row["away_team"]
            w_score, l_score = int(row["home-score"]), int(row["away-score"])
        else:
            winner, loser = row["away_team"], row["home_team"]
            w_score, l_score = int(row["away-score"]), int(row["home-score"])
        labels.append(f"{winner} {w_score}-{l_score} {loser} ({int(row['season'])})")

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"

    fig_height = max(6, len(upsets) * 0.4)
    fig, ax = plt.subplots(figsize=(11, fig_height), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    y_pos = range(len(upsets))
    bars = ax.barh(
        y_pos, upsets["abs_diff"].values,
        color="#e74c3c", alpha=0.85, edgecolor="#333333", linewidth=0.5, height=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=text_color)
    ax.invert_yaxis()

    # Value labels on bars
    for bar, val in zip(bars, upsets["abs_diff"].values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color="#aaaaaa")

    # Axes styling
    ax.set_xlabel("Composite PFF Grade Differential", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(axis="x", colors="#888888", labelsize=9)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    seasons = upsets["season"].unique()
    min_s, max_s = int(min(seasons)), int(max(seasons))
    season_label = f"{min_s}\u2013{max_s}" if min_s != max_s else str(min_s)

    # Title and subtitle
    fig.suptitle(
        "Biggest PFF Grade-Differential Upsets",
        fontsize=15, fontweight="bold", color="white", y=0.98,
    )
    ax.set_title(
        f"{season_label} \u00b7 lower-graded team won \u00b7 top {len(upsets)} by grade gap",
        fontsize=10, color="#888888", pad=14,
    )

    # Footer
    fig.text(
        0.5, 0.002,
        "Source: PFF grades \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    config.GRADES_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.GRADE_DIFFERENTIAL_UPSETS_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.GRADE_DIFFERENTIAL_UPSETS_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_grade_differential_upsets()
