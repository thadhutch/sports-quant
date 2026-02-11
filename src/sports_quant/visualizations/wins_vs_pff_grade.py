"""Generate wins vs composite PFF grade scatter plot."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sports_quant import _config as config

logger = logging.getLogger(__name__)


def generate_wins_vs_pff_grade():
    """Generate and save a scatter plot of win % vs composite PFF grade."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games with no prior PFF data
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()

    # Collect per-team-per-game observations from home + away appearances
    records = []
    for _, row in df.iterrows():
        records.append({
            "team": row["home_team"],
            "season": row["season"],
            "composite": (row["home-off-avg"] + row["home-def-avg"]) / 2,
            "won": row["home-score"] > row["away-score"],
            "lost": row["home-score"] < row["away-score"],
        })
        records.append({
            "team": row["away_team"],
            "season": row["season"],
            "composite": (row["away-off-avg"] + row["away-def-avg"]) / 2,
            "won": row["away-score"] > row["home-score"],
            "lost": row["away-score"] < row["home-score"],
        })

    team_df = pd.DataFrame(records)

    # Aggregate to per-team-per-season averages
    agg = (
        team_df.groupby(["team", "season"])
        .agg(
            composite=("composite", "mean"),
            wins=("won", "sum"),
            losses=("lost", "sum"),
        )
        .reset_index()
    )
    agg["games"] = agg["wins"] + agg["losses"]
    agg["win_pct"] = agg["wins"] / agg["games"]

    n_obs = len(agg)
    seasons = agg["season"].unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)

    # OLS regression
    mask = np.isfinite(agg["composite"]) & np.isfinite(agg["win_pct"])
    slope, intercept, r_value, _, _ = stats.linregress(
        agg.loc[mask, "composite"], agg.loc[mask, "win_pct"]
    )

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Color by season using a continuous colormap
    scatter = ax.scatter(
        agg["composite"], agg["win_pct"],
        c=agg["season"], cmap="plasma", s=40, alpha=0.7,
        edgecolors="#333333", linewidths=0.5,
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=30)
    cbar.set_label("Season", color=text_color, fontsize=10)
    cbar.ax.tick_params(colors=text_color, labelsize=8)
    cbar.outline.set_edgecolor("#333333")

    # Regression line
    x_range = np.array([agg["composite"].min(), agg["composite"].max()])
    ax.plot(x_range, intercept + slope * x_range, color="#4fc3f7", linewidth=1.5,
            label=f"OLS (r={r_value:.2f})")

    # Reference line at .500
    ax.axhline(y=0.5, color="#444444", linewidth=0.8, linestyle="--")

    # Axes styling
    ax.set_xlabel("Avg Composite PFF Grade (off + def / 2)", fontsize=10,
                  color=text_color, labelpad=8)
    ax.set_ylabel("Season Win Percentage", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e", edgecolor="#333333",
              labelcolor=text_color)

    # Title and subtitle
    fig.suptitle(
        "Win Percentage vs Composite PFF Grade",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"{season_label} \u00b7 {n_obs} team-seasons \u00b7 color = season",
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
        config.WINS_VS_PFF_GRADE_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.WINS_VS_PFF_GRADE_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_wins_vs_pff_grade()
