"""Generate PFF grade differential vs Vegas spread scatter plot."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nfl_data_pipeline import _config as config

logger = logging.getLogger(__name__)


def _parse_spread_signed(row: pd.Series) -> float:
    """Extract a signed spread from the Vegas Line, relative to the home team.

    Positive = home team favored, negative = away team favored.
    """
    vl = row["Vegas Line"].strip()
    if vl == "Pick":
        return 0.0
    parts = vl.rsplit(" ", 1)
    spread_val = float(parts[-1])
    favorite_name = parts[0]
    # spread_val is always negative in the raw string (e.g. "Chiefs -7.0")
    if favorite_name == row["home_team"]:
        return abs(spread_val)  # home favored → positive
    return -abs(spread_val)  # away favored → negative


def _compute_pff_differential(df: pd.DataFrame) -> pd.Series:
    """Compute a composite PFF grade differential (home advantage - away advantage).

    Uses the core offensive and defensive grades to create a single
    home-team-relative advantage metric:
        (home_offense - away_defense) + (home_defense - away_offense)
    This simplifies to:
        (home_off + home_def) - (away_off + away_def)
    """
    home_composite = df["home-off-avg"] + df["home-def-avg"]
    away_composite = df["away-off-avg"] + df["away-def-avg"]
    return home_composite - away_composite


def generate_pff_vs_vegas_spread():
    """Generate and save the PFF grade differential vs Vegas spread chart."""
    df = pd.read_csv(config.OVERUNDER_RANKED)
    logger.info("Loaded %d games from %s", len(df), config.OVERUNDER_RANKED)

    # Filter out week-1 games with no prior PFF data
    df = df[(df["home_gp"] > 0) & (df["away_gp"] > 0)].copy()
    df = df[df["Vegas Line"].notna()].copy()
    logger.info("After filtering: %d games", len(df))

    df["spread_signed"] = df.apply(_parse_spread_signed, axis=1)
    df["pff_diff"] = _compute_pff_differential(df)

    # Determine winner for coloring
    df["home_won"] = df["home-score"] > df["away-score"]
    df["away_won"] = df["away-score"] > df["home-score"]

    seasons = df["season"].dropna().unique()
    min_season, max_season = int(min(seasons)), int(max(seasons))
    season_label = f"{min_season}\u2013{max_season}" if min_season != max_season else str(min_season)
    n_games = len(df)

    # OLS regression
    mask = np.isfinite(df["pff_diff"]) & np.isfinite(df["spread_signed"])
    slope, intercept, r_value, _, _ = stats.linregress(
        df.loc[mask, "pff_diff"], df.loc[mask, "spread_signed"]
    )

    # --- Render ---
    bg_color = "#0e1117"
    text_color = "#e0e0e0"
    muted_color = "#888888"

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Color by whether PFF and Vegas agree on the favorite
    agree = (df["pff_diff"] > 0) == (df["spread_signed"] > 0)
    ax.scatter(
        df.loc[agree, "pff_diff"], df.loc[agree, "spread_signed"],
        s=12, alpha=0.35, color="#2ecc71", edgecolors="none", rasterized=True,
        label="PFF & Vegas agree",
    )
    ax.scatter(
        df.loc[~agree, "pff_diff"], df.loc[~agree, "spread_signed"],
        s=12, alpha=0.35, color="#e74c3c", edgecolors="none", rasterized=True,
        label="PFF & Vegas disagree",
    )

    # Regression line
    x_range = np.array([df["pff_diff"].min(), df["pff_diff"].max()])
    ax.plot(x_range, intercept + slope * x_range, color="#f1c40f", linewidth=1.5,
            label=f"OLS (r={r_value:.2f})")

    # Reference lines at zero
    ax.axhline(y=0, color="#444444", linewidth=0.8, linestyle="--")
    ax.axvline(x=0, color="#444444", linewidth=0.8, linestyle="--")

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    quad_style = dict(fontsize=8, color="#555555", fontstyle="italic", ha="center", va="center")
    ax.text(xlim[1] * 0.6, ylim[1] * 0.85, "Both favor home", **quad_style)
    ax.text(xlim[0] * 0.6, ylim[0] * 0.85, "Both favor away", **quad_style)
    ax.text(xlim[1] * 0.6, ylim[0] * 0.85, "PFF: home / Vegas: away", **quad_style)
    ax.text(xlim[0] * 0.6, ylim[1] * 0.85, "PFF: away / Vegas: home", **quad_style)

    # Axes styling
    ax.set_xlabel("PFF Grade Differential (home advantage)", fontsize=10, color=text_color, labelpad=8)
    ax.set_ylabel("Vegas Spread (home favored \u2192 positive)", fontsize=10, color=text_color, labelpad=8)
    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e", edgecolor="#333333",
              labelcolor=text_color)

    # Title and subtitle
    fig.suptitle(
        "PFF Grade Differential vs Vegas Spread",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )
    ax.set_title(
        f"Regular season {season_label} \u00b7 {n_games:,} games \u00b7 r = {r_value:.2f}",
        fontsize=10, color=muted_color, pad=14,
    )

    # Footer
    fig.text(
        0.5, 0.005,
        "Source: PFF grades + PFR/Vegas lines \u00b7 r/sportsbetting",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        config.PFF_VS_VEGAS_SPREAD_CHART,
        dpi=200, bbox_inches="tight", facecolor=bg_color,
    )
    logger.info("Chart saved to %s", config.PFF_VS_VEGAS_SPREAD_CHART)
    plt.close(fig)


if __name__ == "__main__":
    generate_pff_vs_vegas_spread()
