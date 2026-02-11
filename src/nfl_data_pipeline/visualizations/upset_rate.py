"""Generate upset rate by spread size bar chart."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)


def _parse_spread(vegas_line: str) -> float:
    """Extract the numeric spread from a Vegas Line string.

    Example: 'New England Patriots -7.0' -> 7.0 (absolute value)
    Returns 0.0 for pick'em games ('Pick').
    """
    if vegas_line.strip() == "Pick":
        return 0.0
    return abs(float(vegas_line.rsplit(" ", 1)[-1]))


def _determine_underdog(row: pd.Series, favorite_team: str) -> str:
    """Return the underdog team name given the favorite."""
    if favorite_team == row["home_team"]:
        return row["away_team"]
    return row["home_team"]


def _determine_winner(row: pd.Series) -> str | None:
    """Return the winning team, or None for ties."""
    if row["home-score"] > row["away-score"]:
        return row["home_team"]
    elif row["away-score"] > row["home-score"]:
        return row["away_team"]
    return None


def _assign_bucket(spread: float) -> str | None:
    """Assign a spread value to a bucket. Returns None for pick'ems."""
    if spread == 0:
        return None
    if spread <= 3:
        return "1–3"
    if spread <= 7:
        return "3.5–7"
    if spread <= 10:
        return "7.5–10"
    return "10+"


def generate_upset_rate_chart():
    """Generate and save the upset rate by spread size chart."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Parse spread and favorite team from Vegas Line
    df["spread"] = df["Vegas Line"].apply(_parse_spread)
    df["favorite_team"] = df["Vegas Line"].apply(lambda vl: vl.rsplit(" ", 1)[0])

    # Determine underdog and winner
    df["underdog"] = df.apply(lambda r: _determine_underdog(r, r["favorite_team"]), axis=1)
    df["winner"] = df.apply(_determine_winner, axis=1)

    # Exclude ties and pick'ems
    df = df[df["winner"].notna() & (df["spread"] != 0)].copy()

    # Flag upsets (underdog wins straight-up)
    df["upset"] = df["winner"] == df["underdog"]

    # Assign spread buckets
    df["bucket"] = df["spread"].apply(_assign_bucket)
    df = df[df["bucket"].notna()]

    # Calculate upset rate per bucket
    bucket_order = ["1–3", "3.5–7", "7.5–10", "10+"]
    stats = (
        df.groupby("bucket")
        .agg(upsets=("upset", "sum"), games=("upset", "count"))
        .reindex(bucket_order)
    )
    stats["upset_pct"] = (stats["upsets"] / stats["games"] * 100).round(1)

    logger.info("Upset rates:\n%s", stats)

    # Determine season range for subtitle
    seasons = df["season"].dropna().unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}–{max_season}" if min_season != max_season else str(min_season)

    # Build the chart — dark theme styled for Reddit
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    bar_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    bucket_labels = ["1–3 pts", "3.5–7 pts", "7.5–10 pts", "10+ pts"]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    bars = ax.bar(
        range(len(stats)),
        stats["upset_pct"],
        color=bar_colors[: len(stats)],
        width=0.6,
    )

    # Bold percentage labels inside bars
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{row['upset_pct']:.1f}%",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="white",
        )
        # Sample size above bars
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={int(row['games'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#888888",
        )

    # 50% coin-flip reference line
    ax.axhline(y=50, color="#888888", linestyle="--", linewidth=0.8)
    ax.text(
        len(stats) - 0.5,
        51,
        "Coin Flip",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#888888",
        fontstyle="italic",
    )

    # Axes styling
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(bucket_labels, fontsize=11, color=text_color)
    ax.set_yticks([])
    ax.set_ylim(0, max(stats["upset_pct"].max() + 10, 55))
    ax.set_xlabel("Spread Size", fontsize=11, color=text_color, labelpad=8)

    # Remove spines, add subtle gridlines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#333333", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=text_color, length=0)

    # Title and subtitle
    fig.suptitle(
        "NFL Underdog Straight-Up Win Rate by Spread Size",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax.set_title(
        f"Regular season {season_label} · {int(stats['games'].sum()):,} games",
        fontsize=10,
        color="#888888",
        pad=14,
    )

    # Footer with source attribution
    fig.text(
        0.5,
        0.01,
        "Source: PFR/Vegas lines · r/sportsbetting",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    config.LINE_ANALYSIS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.UPSET_RATE_CHART,
        dpi=200,
        bbox_inches="tight",
        facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.UPSET_RATE_CHART)

    plt.show()


if __name__ == "__main__":
    generate_upset_rate_chart()
