"""Generate upset rate by spread size bar chart."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

    # Build the chart
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        stats.index,
        stats["upset_pct"],
        color=sns.color_palette("Blues_d", len(stats)),
        edgecolor="black",
        linewidth=0.5,
    )

    # Annotate bars with game counts
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"n={int(row['games'])}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Spread (points)", fontsize=12)
    ax.set_ylabel("Upset Win %", fontsize=12)
    fig.suptitle("NFL Underdog Straight-Up Win Rate by Spread Size", fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(
        f"Regular season {season_label} · {int(stats['games'].sum()):,} games",
        fontsize=10,
        color="gray",
        pad=12,
    )
    ax.set_ylim(0, stats["upset_pct"].max() + 10)

    plt.tight_layout()

    # Save
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.UPSET_RATE_CHART, dpi=150, bbox_inches="tight")
    logger.info("Chart saved to %s", config.UPSET_RATE_CHART)

    plt.show()


if __name__ == "__main__":
    generate_upset_rate_chart()
