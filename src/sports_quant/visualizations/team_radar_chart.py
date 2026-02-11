"""Generate team radar chart showing PFF grades across all 11 categories."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sports_quant import _config as config

logger = logging.getLogger(__name__)

_CATEGORIES = [
    ("off-avg", "Offense"),
    ("pass-avg", "Passing"),
    ("pblk-avg", "Pass Block"),
    ("recv-avg", "Receiving"),
    ("run-avg", "Rushing"),
    ("rblk-avg", "Run Block"),
    ("def-avg", "Defense"),
    ("rdef-avg", "Run Defense"),
    ("tack-avg", "Tackling"),
    ("prsh-avg", "Pass Rush"),
    ("cov-avg", "Coverage"),
]


def generate_team_radar_chart(team: str | None = None, season: int | None = None):
    """Generate and save a spider/radar chart for one team's PFF grade profile.

    Args:
        team: Team abbreviation (e.g. "KC"). If None, picks the top-ranked team.
        season: Season year. If None, uses the most recent season.
    """
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games with no prior PFF data
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    if season is None:
        season = int(df["season"].max())
    df = df[df["season"] == season].copy()
    logger.info("Season %d: %d games", season, len(df))

    # Collect per-team grade observations from home + away appearances
    records = []
    for _, row in df.iterrows():
        for suffix, _ in _CATEGORIES:
            records.append({
                "team": row["home_team"],
                "category": suffix,
                "grade": row[f"home-{suffix}"],
            })
            records.append({
                "team": row["away_team"],
                "category": suffix,
                "grade": row[f"away-{suffix}"],
            })

    grades_df = pd.DataFrame(records)
    avg_grades = grades_df.groupby(["team", "category"])["grade"].mean().unstack("category")

    # Pick team: default to best overall average grade
    if team is None:
        team = avg_grades.mean(axis=1).idxmax()
    if team not in avg_grades.index:
        raise ValueError(f"Team '{team}' not found in season {season}")

    team_grades = avg_grades.loc[team]

    # League averages for reference
    league_avg = avg_grades.mean()

    # Order values to match _CATEGORIES
    suffixes = [s for s, _ in _CATEGORIES]
    labels = [d for _, d in _CATEGORIES]
    values = [team_grades[s] for s in suffixes]
    league_values = [league_avg[s] for s in suffixes]

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    # Close the polygon
    values += values[:1]
    league_values += league_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Team polygon
    ax.plot(angles, values, "o-", color="#4fc3f7", linewidth=2, markersize=5, label=team)
    ax.fill(angles, values, alpha=0.20, color="#4fc3f7")

    # League average polygon
    ax.plot(angles, league_values, "o--", color="#888888", linewidth=1.2, markersize=3,
            label="League Avg")
    ax.fill(angles, league_values, alpha=0.08, color="#888888")

    # Spoke labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color=text_color)

    # Radial grid styling
    ax.tick_params(axis="y", labelsize=7, colors="#666666")
    ax.set_rlabel_position(30)
    ax.yaxis.grid(True, color="#333333", linewidth=0.5)
    ax.xaxis.grid(True, color="#333333", linewidth=0.5)
    ax.spines["polar"].set_color("#333333")

    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.12), fontsize=9,
              facecolor="#1a1a2e", edgecolor="#333333", labelcolor=text_color)

    # Title and subtitle
    fig.suptitle(
        f"{team} PFF Grade Profile",
        fontsize=15, fontweight="bold", color="white", y=0.98,
    )
    ax.set_title(
        f"{season} Regular Season \u00b7 11 PFF categories \u00b7 vs league average",
        fontsize=10, color="#888888", pad=24,
    )

    # Footer
    fig.text(
        0.5, 0.01,
        "Source: PFF grades \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    config.TEAMS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.TEAM_RADAR_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.TEAM_RADAR_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_team_radar_chart()
