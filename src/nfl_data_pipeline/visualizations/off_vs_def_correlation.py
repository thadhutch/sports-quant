"""Generate offensive vs defensive grade correlation scatter plot."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)


def generate_off_vs_def_correlation():
    """Generate and save the offensive vs defensive grade correlation chart."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games with no prior PFF data
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()
    logger.info("After filtering: %d games", len(df))

    # Collect per-team-per-game observations (each team appears as home and away)
    records = []
    for _, row in df.iterrows():
        records.append({
            "team": row["home_team"],
            "season": row["season"],
            "off_grade": row["home-off-avg"],
            "def_grade": row["home-def-avg"],
            "won": row["home-score"] > row["away-score"],
            "lost": row["home-score"] < row["away-score"],
        })
        records.append({
            "team": row["away_team"],
            "season": row["season"],
            "off_grade": row["away-off-avg"],
            "def_grade": row["away-def-avg"],
            "won": row["away-score"] > row["home-score"],
            "lost": row["away-score"] < row["home-score"],
        })

    team_df = pd.DataFrame(records)

    # Aggregate to per-team-per-season averages
    agg = (
        team_df.groupby(["team", "season"])
        .agg(
            off_grade=("off_grade", "mean"),
            def_grade=("def_grade", "mean"),
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
    mask = np.isfinite(agg["off_grade"]) & np.isfinite(agg["def_grade"])
    slope, intercept, r_value, _, _ = stats.linregress(
        agg.loc[mask, "off_grade"], agg.loc[mask, "def_grade"]
    )

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Color by win percentage using a continuous colormap
    scatter = ax.scatter(
        agg["off_grade"], agg["def_grade"],
        c=agg["win_pct"], cmap="RdYlGn", vmin=0, vmax=1,
        s=40, alpha=0.7, edgecolors="#333333", linewidths=0.5,
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=30)
    cbar.set_label("Win %", color=text_color, fontsize=10)
    cbar.ax.tick_params(colors=text_color, labelsize=8)
    cbar.outline.set_edgecolor("#333333")

    # Regression line
    x_range = np.array([agg["off_grade"].min(), agg["off_grade"].max()])
    ax.plot(x_range, intercept + slope * x_range, color="#4fc3f7", linewidth=1.5,
            label=f"OLS (r={r_value:.2f})")

    # Quadrant lines at median values
    off_med = agg["off_grade"].median()
    def_med = agg["def_grade"].median()
    ax.axhline(y=def_med, color="#444444", linewidth=0.8, linestyle="--")
    ax.axvline(x=off_med, color="#444444", linewidth=0.8, linestyle="--")

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    quad_style = dict(fontsize=8, color="#555555", fontstyle="italic", ha="center", va="center")
    ax.text(xlim[1] * 0.97, ylim[1] * 0.97, "Elite both sides",
            ha="right", va="top", fontsize=8, color="#2ecc71", fontstyle="italic")
    ax.text(xlim[0] * 1.01, ylim[0] * 1.01, "Weak both sides",
            ha="left", va="bottom", fontsize=8, color="#e74c3c", fontstyle="italic")
    ax.text(xlim[1] * 0.97, ylim[0] * 1.01, "Offense carries",
            ha="right", va="bottom", fontsize=8, color="#f1c40f", fontstyle="italic")
    ax.text(xlim[0] * 1.01, ylim[1] * 0.97, "Defense carries",
            ha="left", va="top", fontsize=8, color="#f1c40f", fontstyle="italic")

    # Axes styling
    ax.set_xlabel("Offensive PFF Grade (season avg)", fontsize=10, color=text_color, labelpad=8)
    ax.set_ylabel("Defensive PFF Grade (season avg)", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(loc="lower right", fontsize=9, facecolor="#1a1a2e", edgecolor="#333333",
              labelcolor=text_color)

    # Title and subtitle
    fig.suptitle(
        "Offensive vs Defensive PFF Grade by Team-Season",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"{season_label} \u00b7 {n_obs} team-seasons \u00b7 color = win %",
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
        config.OFF_VS_DEF_CORRELATION_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.OFF_VS_DEF_CORRELATION_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_off_vs_def_correlation()
