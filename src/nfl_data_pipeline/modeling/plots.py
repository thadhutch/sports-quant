"""All modeling-related charts (training + backtesting)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark theme constants (matches repo-wide chart style)
# ---------------------------------------------------------------------------
_BG = "#0e1117"
_TEXT = "#e0e0e0"
_SUBTITLE = "#888888"
_FOOTER = "#555555"
_GRID = "#333333"

_CYAN = "#4fc3f7"
_CORAL = "#ff7043"
_GREEN = "#2ecc71"
_GOLD = "#f1c40f"


def _style_ax(ax: plt.Axes) -> None:
    """Apply the shared dark-theme styling to an axes object."""
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=_TEXT, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=_TEXT, labelsize=9, length=0)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    logger.info("Saved plot to %s", path)


# ---------------------------------------------------------------------------
# Training plots (ported from nfl-model/algorithm.py)
# ---------------------------------------------------------------------------


def plot_accuracy_by_confidence(
    accuracy_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Estimated vs actual accuracy and prediction counts by confidence bin."""
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor=_BG)
    _style_ax(ax1)

    total_picks = int(accuracy_df["Prediction_Count"].sum())

    # Gray bars for prediction count on twin axis (draw first so lines sit on top)
    ax2 = ax1.twinx()
    ax2.set_facecolor(_BG)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.bar(
        accuracy_df["Confidence Bin"],
        accuracy_df["Prediction_Count"],
        alpha=0.25,
        color=_TEXT,
        label="Prediction Count",
        zorder=1,
    )
    ax2.set_ylabel("Prediction Count", fontsize=10, color=_SUBTITLE)
    ax2.tick_params(axis="y", colors=_SUBTITLE, labelsize=8, length=0)

    # Cyan line: estimated accuracy
    ax1.plot(
        accuracy_df["Confidence Bin"],
        accuracy_df["Average_Adjusted_Score"],
        marker="o",
        markersize=6,
        label="Estimated Accuracy",
        color=_CYAN,
        linewidth=2,
        zorder=3,
    )
    # Green line: actual accuracy
    ax1.plot(
        accuracy_df["Confidence Bin"],
        accuracy_df["Actual_Accuracy"],
        marker="o",
        markersize=6,
        label="Actual Accuracy",
        color=_GREEN,
        linewidth=2,
        zorder=3,
    )

    # Gold 50% reference line
    ax1.axhline(y=0.5, color=_GOLD, linestyle="--", linewidth=0.8, zorder=2)
    ax1.text(
        len(accuracy_df) - 1,
        0.505,
        "Coin Flip",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_GOLD,
        fontstyle="italic",
    )

    ax1.set_xlabel("Confidence Interval", fontsize=10, color=_TEXT)
    ax1.set_ylabel("Accuracy", fontsize=10, color=_TEXT)
    ax1.set_xticks(range(len(accuracy_df)))
    ax1.set_xticklabels(accuracy_df["Confidence Bin"], rotation=45, ha="right")
    ax1.legend(
        loc="upper left",
        fontsize=9,
        facecolor="#1a1a2e",
        edgecolor=_GRID,
        labelcolor=_TEXT,
    )

    fig.suptitle(
        "Estimated vs Actual Accuracy by Confidence",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax1.set_title(
        f"{total_picks:,} total picks",
        fontsize=10,
        color=_SUBTITLE,
        pad=14,
    )
    fig.text(
        0.5, 0.01,
        "Source: PFF grades + PFR/Vegas lines \u00b7 50-model ensemble",
        ha="center", fontsize=8, color=_FOOTER,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out_path)


def plot_accuracy_by_algorithm_score(
    accuracy_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Actual accuracy and counts by algorithm score bin (overall)."""
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor=_BG)
    _style_ax(ax1)

    total_picks = int(accuracy_df["Prediction_Count"].sum())

    # Gray bars for prediction count on twin axis
    ax2 = ax1.twinx()
    ax2.set_facecolor(_BG)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.bar(
        accuracy_df["Algorithm Score Bin"],
        accuracy_df["Prediction_Count"],
        alpha=0.25,
        color=_TEXT,
        label="Prediction Count",
        zorder=1,
    )
    ax2.set_ylabel("Prediction Count", fontsize=10, color=_SUBTITLE)
    ax2.tick_params(axis="y", colors=_SUBTITLE, labelsize=8, length=0)

    # Cyan line: actual accuracy
    ax1.plot(
        accuracy_df["Algorithm Score Bin"],
        accuracy_df["Actual_Accuracy"],
        marker="o",
        markersize=6,
        label="Actual Accuracy",
        color=_CYAN,
        linewidth=2,
        zorder=3,
    )

    # Gold 50% reference line
    ax1.axhline(y=0.5, color=_GOLD, linestyle="--", linewidth=0.8, zorder=2)
    ax1.text(
        len(accuracy_df) - 1,
        0.505,
        "Coin Flip",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_GOLD,
        fontstyle="italic",
    )

    ax1.set_xlabel("Algorithm Score Bin", fontsize=10, color=_TEXT)
    ax1.set_ylabel("Actual Accuracy", fontsize=10, color=_TEXT)
    ax1.set_xticks(range(len(accuracy_df)))
    ax1.set_xticklabels(accuracy_df["Algorithm Score Bin"], rotation=45, ha="right")
    ax1.legend(
        loc="upper left",
        fontsize=9,
        facecolor="#1a1a2e",
        edgecolor=_GRID,
        labelcolor=_TEXT,
    )

    fig.suptitle(
        "Accuracy by Algorithm Score",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax1.set_title(
        f"{total_picks:,} total picks",
        fontsize=10,
        color=_SUBTITLE,
        pad=14,
    )
    fig.text(
        0.5, 0.01,
        "Source: PFF grades + PFR/Vegas lines \u00b7 50-model ensemble",
        ha="center", fontsize=8, color=_FOOTER,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out_path)


def plot_accuracy_by_algorithm_score_season(
    accuracy_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Actual accuracy by algorithm score bin, split by season — heatmap."""
    pivot = accuracy_df.pivot(
        index="Algorithm Score Bin", columns="Season", values="Actual_Accuracy"
    )
    # Drop rows that are entirely NaN (empty bins)
    pivot = pivot.dropna(how="all")
    # Convert season columns to int labels
    pivot.columns = [str(int(c)) for c in pivot.columns]

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.4), max(6, len(pivot) * 0.5)),
        facecolor=_BG,
    )
    ax.set_facecolor(_BG)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor=_GRID,
        square=False,
        ax=ax,
        annot_kws={"size": 8, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8, "aspect": 30},
    )

    # Style colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=_TEXT, labelsize=8)
    cbar.outline.set_edgecolor(_GRID)

    ax.set_xlabel("Season", fontsize=10, color=_TEXT)
    ax.set_ylabel("Algorithm Score Bin", fontsize=10, color=_TEXT)
    ax.tick_params(axis="x", colors=_TEXT, labelsize=9, length=0, rotation=0)
    ax.tick_params(axis="y", colors=_TEXT, labelsize=8, length=0, rotation=0)

    fig.suptitle(
        "Accuracy by Algorithm Score & Season",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax.set_title(
        "Prediction accuracy per bin \u00b7 darker red = lower, darker green = higher",
        fontsize=10,
        color=_SUBTITLE,
        pad=14,
    )
    fig.text(
        0.5, 0.01,
        "Source: PFF grades + PFR/Vegas lines \u00b7 50-model ensemble",
        ha="center", fontsize=8, color=_FOOTER,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out_path)


def plot_cumulative_profit(
    picks_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Cumulative profit in units over time."""
    df = picks_df.sort_values("Date").copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=_BG)
    _style_ax(ax)

    # Cyan line + filled area for units profit
    ax.plot(
        df["Date"],
        df["Cumulative Profit (Units)"],
        color=_CYAN,
        linewidth=1.8,
        zorder=3,
    )
    ax.fill_between(
        df["Date"],
        df["Cumulative Profit (Units)"],
        alpha=0.15,
        color=_CYAN,
        zorder=2,
    )
    ax.set_ylabel("Cumulative Profit (Units)", fontsize=10, color=_TEXT)

    # Season boundary vertical lines
    if "Season" in df.columns:
        seasons = df["Season"].unique()
        for s in seasons[1:]:
            first_date = df.loc[df["Season"] == s, "Date"].min()
            ax.axvline(
                x=first_date, color=_GRID, linestyle="--", linewidth=0.8, zorder=1,
            )

    # Annotate final value
    final_units = df["Cumulative Profit (Units)"].iloc[-1]
    last_date = df["Date"].iloc[-1]
    ax.annotate(
        f"{final_units:+.1f} u",
        xy=(last_date, final_units),
        fontsize=10,
        fontweight="bold",
        color=_CYAN,
        ha="left",
        va="bottom",
        xytext=(8, 4),
        textcoords="offset points",
    )

    ax.set_xlabel("Date", fontsize=10, color=_TEXT)

    fig.suptitle(
        "Cumulative Profit Over Time",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )

    # Build subtitle with date range
    min_date = df["Date"].min().strftime("%b %Y")
    max_date = df["Date"].max().strftime("%b %Y")
    ax.set_title(
        f"{min_date} \u2013 {max_date} \u00b7 {len(df):,} picks",
        fontsize=10,
        color=_SUBTITLE,
        pad=14,
    )
    fig.text(
        0.5, 0.01,
        "Source: PFF grades + PFR/Vegas lines \u00b7 50-model ensemble",
        ha="center", fontsize=8, color=_FOOTER,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Backtest plots (ported from nfl-model/backtest.py)
# ---------------------------------------------------------------------------


def plot_accuracy_by_season(
    avg_season_accuracy: pd.DataFrame,
    n_models: int,
    out_path: Path,
) -> None:
    """Average backtest accuracy by season — gradient bars."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG)
    _style_ax(ax)

    seasons = avg_season_accuracy["Season"].astype(str)
    accuracy = avg_season_accuracy["Accuracy"]
    overall_avg = accuracy.mean()

    # Gradient bar colors from RdYlGn
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=accuracy.min() - 0.02, vmax=accuracy.max() + 0.02)
    colors = [cmap(norm(v)) for v in accuracy]

    bars = ax.bar(seasons, accuracy, color=colors, width=0.6, zorder=3)

    # Bold white percentage labels inside bars
    for bar, val in zip(bars, accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.1%}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

    # Gold dashed overall average line
    ax.axhline(y=overall_avg, color=_GOLD, linestyle="--", linewidth=0.8, zorder=2)
    ax.text(
        len(seasons) - 0.5,
        overall_avg + 0.003,
        f"Avg {overall_avg:.1%}",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_GOLD,
        fontstyle="italic",
    )

    ax.set_xlabel("Season", fontsize=11, color=_TEXT, labelpad=8)
    ax.set_yticks([])
    ax.set_ylim(0, accuracy.max() + 0.08)

    fig.suptitle(
        "Average Model Accuracy by Season",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax.set_title(
        f"Walk-forward backtest \u00b7 {n_models} models",
        fontsize=10,
        color=_SUBTITLE,
        pad=14,
    )
    fig.text(
        0.5, 0.01,
        "Source: PFF grades + PFR/Vegas lines \u00b7 50-model ensemble",
        ha="center", fontsize=8, color=_FOOTER,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out_path)


def plot_accuracy_by_confidence_interval(
    avg_confidence_accuracy: pd.DataFrame,
    n_models: int,
    out_path: Path,
) -> None:
    """Average backtest accuracy by confidence interval — gradient bars."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
    _style_ax(ax)

    bins = avg_confidence_accuracy["Confidence Bin"].astype(str)
    accuracy = avg_confidence_accuracy["Accuracy"]

    # Gradient bar colors from RdYlGn
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=accuracy.min() - 0.02, vmax=accuracy.max() + 0.02)
    colors = [cmap(norm(v)) for v in accuracy]

    bars = ax.bar(bins, accuracy, color=colors, width=0.6, zorder=3)

    # Bold white percentage labels inside bars
    for bar, val in zip(bars, accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.1%}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

    # Gold 50% coin-flip reference line
    ax.axhline(y=0.5, color=_GOLD, linestyle="--", linewidth=0.8, zorder=2)
    ax.text(
        len(bins) - 0.5,
        0.503,
        "Coin Flip",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_GOLD,
        fontstyle="italic",
    )

    ax.set_xlabel("Confidence Interval", fontsize=11, color=_TEXT, labelpad=8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_ylim(0, accuracy.max() + 0.08)

    fig.suptitle(
        "Average Model Accuracy by Confidence",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax.set_title(
        f"Walk-forward backtest \u00b7 {n_models} models",
        fontsize=10,
        color=_SUBTITLE,
        pad=14,
    )
    fig.text(
        0.5, 0.01,
        "Source: PFF grades + PFR/Vegas lines \u00b7 50-model ensemble",
        ha="center", fontsize=8, color=_FOOTER,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out_path)


def plot_accuracy_by_confidence_and_season(
    avg_conf_season: pd.DataFrame,
    n_models: int,
    out_path: Path,
) -> None:
    """Average backtest accuracy by confidence interval and season — heatmap."""
    pivot = avg_conf_season.pivot(
        index="Confidence Bin", columns="Season", values="Accuracy"
    )
    # Drop rows that are entirely NaN
    pivot = pivot.dropna(how="all")
    # Convert season columns to int labels
    pivot.columns = [str(int(c)) for c in pivot.columns]

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.4), max(6, len(pivot) * 0.6)),
        facecolor=_BG,
    )
    ax.set_facecolor(_BG)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=0.55,
        linewidths=0.5,
        linecolor=_GRID,
        square=False,
        ax=ax,
        annot_kws={"size": 8, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8, "aspect": 30},
    )

    # Style colorbar text
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=_TEXT, labelsize=8)
    cbar.outline.set_edgecolor(_GRID)

    ax.set_xlabel("Season", fontsize=10, color=_TEXT)
    ax.set_ylabel("Confidence Bin", fontsize=10, color=_TEXT)
    ax.tick_params(axis="x", colors=_TEXT, labelsize=9, length=0, rotation=0)
    ax.tick_params(axis="y", colors=_TEXT, labelsize=8, length=0, rotation=0)

    fig.suptitle(
        "Accuracy by Confidence & Season",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    ax.set_title(
        f"Walk-forward backtest \u00b7 {n_models} models",
        fontsize=10,
        color=_SUBTITLE,
        pad=14,
    )
    fig.text(
        0.5, 0.01,
        "Source: PFF grades + PFR/Vegas lines \u00b7 50-model ensemble",
        ha="center", fontsize=8, color=_FOOTER,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out_path)
