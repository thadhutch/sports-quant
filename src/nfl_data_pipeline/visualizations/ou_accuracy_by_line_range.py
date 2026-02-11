"""Generate O/U accuracy by line range bar chart."""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)


def _assign_ou_bucket(ou_line: float) -> str | None:
    """Assign an O/U line value to a bucket."""
    if ou_line < 38:
        return "<38"
    if ou_line < 42:
        return "38\u201341.5"
    if ou_line < 46:
        return "42\u201345.5"
    if ou_line < 50:
        return "46\u201349.5"
    return "50+"


def generate_ou_accuracy_by_line_range():
    """Generate and save the O/U accuracy by line range chart."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out pushes/no-data
    df = df[df["total"] != 2].copy()
    df = df[df["ou_line"].notna()].copy()
    logger.info("After filtering: %d games", len(df))

    df["actual_total"] = df["home-score"] + df["away-score"]
    df["went_over"] = (df["actual_total"] > df["ou_line"]).astype(int)
    df["bucket"] = df["ou_line"].apply(_assign_ou_bucket)
    df = df[df["bucket"].notna()]

    bucket_order = ["<38", "38\u201341.5", "42\u201345.5", "46\u201349.5", "50+"]
    stats = (
        df.groupby("bucket")
        .agg(overs=("went_over", "sum"), games=("went_over", "count"))
        .reindex(bucket_order)
    )
    stats["over_pct"] = (stats["overs"] / stats["games"] * 100).round(1)

    seasons = df["season"].dropna().unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    n_games = int(stats["games"].sum())

    logger.info("Over rates by bucket:\n%s", stats)

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    bar_colors = ["#4fc3f7", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    bars = ax.bar(
        range(len(stats)), stats["over_pct"],
        color=bar_colors[: len(stats)], width=0.6,
    )

    # Percentage labels inside bars
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{row['over_pct']:.1f}%",
            ha="center", va="center",
            fontsize=18, fontweight="bold", color="white",
        )
        # Sample size above bars
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={int(row['games'])}",
            ha="center", va="bottom",
            fontsize=9, color="#888888",
        )

    # 50% reference line
    ax.axhline(y=50, color="#888888", linestyle="--", linewidth=0.8)
    ax.text(
        len(stats) - 0.5, 51, "50%",
        ha="right", va="bottom", fontsize=9, color="#888888", fontstyle="italic",
    )

    # Axes styling
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(bucket_order, fontsize=11, color=text_color)
    ax.set_yticks([])
    ax.set_ylim(0, max(stats["over_pct"].max() + 10, 55))
    ax.set_xlabel("O/U Line Range", fontsize=11, color=text_color, labelpad=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#333333", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=text_color, length=0)

    # Title and subtitle
    fig.suptitle(
        "Over Rate by O/U Line Range",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"Regular season {season_label} \u00b7 {n_games:,} games",
        fontsize=10, color="#888888", pad=14,
    )

    # Footer
    fig.text(
        0.5, 0.01,
        "Source: PFR/Vegas lines \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    config.LINE_ANALYSIS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.OU_ACCURACY_BY_RANGE_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.OU_ACCURACY_BY_RANGE_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_ou_accuracy_by_line_range()
