"""All modeling-related charts (training + backtesting)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
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
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Confidence Interval")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(
        accuracy_df["Confidence Bin"],
        accuracy_df["Average_Adjusted_Score"],
        marker="o",
        label="Estimated Accuracy",
        color="tab:blue",
    )
    ax1.plot(
        accuracy_df["Confidence Bin"],
        accuracy_df["Actual_Accuracy"],
        marker="x",
        label="Actual Accuracy",
        color="tab:green",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")
    ax1.set_xticklabels(accuracy_df["Confidence Bin"], rotation=45)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Prediction Count", color="tab:red")
    ax2.bar(
        accuracy_df["Confidence Bin"],
        accuracy_df["Prediction_Count"],
        alpha=0.3,
        color="tab:red",
        label="Prediction Count",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title(
        "Estimated vs Actual Accuracy and Prediction Counts by Confidence Interval"
    )
    plt.tight_layout()
    _save(fig, out_path)


def plot_accuracy_by_algorithm_score(
    accuracy_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Actual accuracy and counts by algorithm score bin (overall)."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Algorithm Score Bin")
    ax1.set_ylabel("Actual Accuracy", color="tab:blue")
    ax1.plot(
        accuracy_df["Algorithm Score Bin"],
        accuracy_df["Actual_Accuracy"],
        marker="o",
        label="Actual Accuracy",
        color="tab:blue",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")
    ax1.set_xticklabels(accuracy_df["Algorithm Score Bin"], rotation=45)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Prediction Count", color="tab:red")
    ax2.bar(
        accuracy_df["Algorithm Score Bin"],
        accuracy_df["Prediction_Count"],
        alpha=0.3,
        color="tab:red",
        label="Prediction Count",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title("Actual Accuracy and Prediction Counts by Algorithm Score Bin")
    plt.tight_layout()
    _save(fig, out_path)


def plot_accuracy_by_algorithm_score_season(
    accuracy_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Actual accuracy by algorithm score bin, split by season."""
    pivot = accuracy_df.pivot(
        index="Algorithm Score Bin", columns="Season", values="Actual_Accuracy"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Actual Accuracy by Algorithm Score Bin and Season")
    ax.set_xlabel("Algorithm Score Bin")
    ax.set_ylabel("Actual Accuracy")
    plt.xticks(rotation=45)
    ax.legend(title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    _save(fig, out_path)


def plot_cumulative_profit(
    picks_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Cumulative profit in units and dollars over time."""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Profit (Units)", color="tab:blue")
    ax1.plot(
        picks_df["Date"],
        picks_df["Cumulative Profit (Units)"],
        marker="o",
        markersize=3,
        label="Units",
        color="tab:blue",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Cumulative Profit ($)", color="tab:red")
    ax2.plot(
        picks_df["Date"],
        picks_df["Cumulative Profit ($)"],
        marker="x",
        markersize=3,
        label="Dollars",
        color="tab:red",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title("Cumulative Profit Over Time")
    plt.grid(True)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Backtest plots (ported from nfl-model/backtest.py)
# ---------------------------------------------------------------------------


def plot_accuracy_by_season(
    avg_season_accuracy: pd.DataFrame,
    n_models: int,
    out_path: Path,
) -> None:
    """Average backtest accuracy by season."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        avg_season_accuracy["Season"],
        avg_season_accuracy["Accuracy"],
        marker="o",
        linestyle="-",
    )
    ax.set_title(f"Average Model Accuracy by Season over {n_models} models")
    ax.set_xlabel("Season")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    plt.tight_layout()
    _save(fig, out_path)


def plot_accuracy_by_confidence_interval(
    avg_confidence_accuracy: pd.DataFrame,
    n_models: int,
    out_path: Path,
) -> None:
    """Average backtest accuracy by confidence interval."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        avg_confidence_accuracy["Confidence Bin"],
        avg_confidence_accuracy["Accuracy"],
        color="skyblue",
    )
    ax.set_title(
        f"Average Model Accuracy by Confidence Interval over {n_models} models"
    )
    ax.set_xlabel("Confidence Interval")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=45)
    ax.grid(axis="y")
    plt.tight_layout()
    _save(fig, out_path)


def plot_accuracy_by_confidence_and_season(
    avg_conf_season: pd.DataFrame,
    n_models: int,
    out_path: Path,
) -> None:
    """Average backtest accuracy by confidence interval and season."""
    pivot = avg_conf_season.pivot(
        index="Confidence Bin", columns="Season", values="Accuracy"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(
        f"Average Model Accuracy by Confidence & Season over {n_models} models"
    )
    ax.set_xlabel("Confidence Interval")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=45)
    ax.legend(title="Season", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y")
    plt.tight_layout()
    _save(fig, out_path)
