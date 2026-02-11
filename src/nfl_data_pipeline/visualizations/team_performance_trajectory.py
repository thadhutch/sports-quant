"""Generate team performance trajectory line chart (rolling PFF grades by game)."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)

# Core PFF grade categories to plot
_GRADE_CATEGORIES = [
    ("off-avg", "Offense", "#2ecc71"),
    ("def-avg", "Defense", "#e74c3c"),
    ("pass-avg", "Passing", "#4fc3f7"),
    ("run-avg", "Rushing", "#f1c40f"),
    ("cov-avg", "Coverage", "#9b59b6"),
]


def _collect_team_games(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Collect all games for a team (home and away) with their PFF grades,
    sorted chronologically by date."""
    home_games = df[df["home_team"] == team].copy()
    away_games = df[df["away_team"] == team].copy()

    records = []
    for _, row in home_games.iterrows():
        record = {"date": row["Formatted Date"], "gp": row["home_gp"]}
        for suffix, _, _ in _GRADE_CATEGORIES:
            record[suffix] = row[f"home-{suffix}"]
        records.append(record)

    for _, row in away_games.iterrows():
        record = {"date": row["Formatted Date"], "gp": row["away_gp"]}
        for suffix, _, _ in _GRADE_CATEGORIES:
            record[suffix] = row[f"away-{suffix}"]
        records.append(record)

    team_df = pd.DataFrame(records)
    team_df = team_df.sort_values("date").reset_index(drop=True)
    team_df["game_num"] = range(1, len(team_df) + 1)
    return team_df


def generate_team_performance_trajectory(
    team: str | None = None, season: int | None = None
):
    """Generate and save the team performance trajectory chart.

    Args:
        team: Team name to visualize. If None, picks the team with the most
              games in the selected season.
        season: Season to visualize. If None, uses the most recent season.
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter to week 2+ (need prior PFF data)
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    if season is None:
        season = int(df["season"].max())
    df = df[df["season"] == season].copy()
    logger.info("Season %d: %d games", season, len(df))

    if team is None:
        # Pick the team with the most game appearances
        all_teams = pd.concat([df["home_team"], df["away_team"]])
        team = all_teams.value_counts().index[0]
    logger.info("Team: %s", team)

    team_df = _collect_team_games(df, team)
    n_games = len(team_df)

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    for suffix, display_name, color in _GRADE_CATEGORIES:
        ax.plot(
            team_df["game_num"], team_df[suffix],
            color=color, linewidth=2, alpha=0.85, label=display_name,
            marker="o", markersize=4,
        )

    # Highlight the average line for reference
    ax.axhline(y=50, color="#444444", linewidth=0.8, linestyle="--")
    ax.text(
        n_games + 0.3, 50, "League avg",
        va="center", fontsize=8, color="#555555", fontstyle="italic",
    )

    # Axes styling
    ax.set_xlabel("Game Number", fontsize=10, color=text_color, labelpad=8)
    ax.set_ylabel("Rolling Avg PFF Grade", fontsize=10, color=text_color, labelpad=8)
    ax.set_xlim(0.5, n_games + 0.5)
    ax.set_xticks(range(1, n_games + 1))
    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(
        loc="upper left", fontsize=9, facecolor="#1a1a2e",
        edgecolor="#333333", labelcolor=text_color,
    )

    # Title and subtitle
    fig.suptitle(
        f"{team} \u2014 Performance Trajectory",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"{season} Regular Season \u00b7 {n_games} games",
        fontsize=10, color=muted_color, pad=14,
    )

    # Footer
    fig.text(
        0.5, 0.005,
        "Source: PFF grades \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.TEAM_TRAJECTORY_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.TEAM_TRAJECTORY_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_team_performance_trajectory()
