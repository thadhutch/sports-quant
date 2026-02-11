"""Financial simulation and performance statistics for O/U betting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_betting(
    picks_df: pd.DataFrame,
    starting_capital: float = 100.0,
) -> pd.DataFrame:
    """Run a 1% Kelly criterion simulation on *picks_df*.

    Bet sizing:
      - 1 unit = 1% of current capital
      - Win: gain 1 unit  (1:1 payout)
      - Loss: lose 1.1 units (1.1:1 vig)

    The input must contain ``Correct Prediction`` (1/0) and ``Date`` columns.
    Returns *picks_df* augmented with profit and cumulative-profit columns,
    sorted by date.
    """
    df = picks_df.sort_values("Date").copy()

    # Unit-based profit
    df["Profit (Units)"] = np.where(
        df["Correct Prediction"] == 1, 1.0, -1.1
    )
    df["Cumulative Profit (Units)"] = df["Profit (Units)"].cumsum()

    # Dollar-based profit (compound via 1% of rolling capital)
    capital = starting_capital
    cum_dollars: list[float] = []
    for correct in df["Correct Prediction"]:
        unit_size = capital * 0.01
        profit = unit_size if correct == 1 else -1.1 * unit_size
        capital += profit
        cum_dollars.append(capital - starting_capital)

    df["Cumulative Profit ($)"] = cum_dollars

    return df


def compute_performance_stats(
    picks_df: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Compute best/worst day/week/month/year in units and dollars.

    *picks_df* must have ``Date``, ``Profit (Units)``, and
    ``Cumulative Profit ($)`` columns (as produced by :func:`simulate_betting`).

    Returns a nested dict suitable for display or writing to a text file.
    """
    df = picks_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    stats: dict[str, dict[str, str]] = {"units": {}, "dollars": {}}

    # --- Units ---
    for freq, label in [("D", "Day"), ("W", "Week"), ("ME", "Month"), ("YE", "Year")]:
        series = df["Profit (Units)"].resample(freq).sum()
        if series.empty:
            continue
        best_idx = series.idxmax()
        worst_idx = series.idxmin()

        if label == "Month":
            fmt = lambda d: d.strftime("%Y-%m")
        elif label == "Year":
            fmt = lambda d: str(d.year)
        else:
            fmt = lambda d: str(d.date())

        stats["units"][f"Best {label}"] = (
            f"{fmt(best_idx)} with profit {series[best_idx]:.2f} units"
        )
        stats["units"][f"Worst {label}"] = (
            f"{fmt(worst_idx)} with profit {series[worst_idx]:.2f} units"
        )

    # --- Dollars ---
    for freq, label in [("D", "Day"), ("W", "Week"), ("ME", "Month"), ("YE", "Year")]:
        series = df["Cumulative Profit ($)"].resample(freq).last()
        if series.empty:
            continue
        best_idx = series.idxmax()
        worst_idx = series.idxmin()

        if label == "Month":
            fmt = lambda d: d.strftime("%Y-%m")
        elif label == "Year":
            fmt = lambda d: str(d.year)
        else:
            fmt = lambda d: str(d.date())

        stats["dollars"][f"Best {label}"] = (
            f"{fmt(best_idx)} with profit ${series[best_idx]:.2f}"
        )
        stats["dollars"][f"Worst {label}"] = (
            f"{fmt(worst_idx)} with profit ${series[worst_idx]:.2f}"
        )

    return stats


def write_performance_stats(stats: dict, path: Path) -> None:
    """Write performance statistics to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("Performance Statistics (Units):\n")
        for k, v in stats.get("units", {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nPerformance Statistics ($):\n")
        for k, v in stats.get("dollars", {}).items():
            f.write(f"  {k}: {v}\n")
    logger.info("Saved performance statistics to %s", path)
