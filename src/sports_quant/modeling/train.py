"""Ensemble training orchestrator.

For each test date (outer loop), trains *N* models (inner loop), filters
to those above an accuracy threshold, selects the top 3 by weighted
seasonal accuracy, requires consensus, then runs a financial simulation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports_quant import _config as config
from sports_quant.modeling._data import DATE_COLUMN, TARGET_COLUMN, load_and_prepare
from sports_quant.modeling._scoring import (
    compute_season_progress,
    compute_weighted_accuracy,
    predict_with_consensus,
    select_top_models,
)
from sports_quant.modeling._simulation import (
    compute_performance_stats,
    simulate_betting,
    write_performance_stats,
)
from sports_quant.modeling._training import train_ensemble_for_date
from sports_quant.modeling.plots import (
    plot_accuracy_by_algorithm_score,
    plot_accuracy_by_algorithm_score_season,
    plot_accuracy_by_confidence,
    plot_cumulative_profit,
)

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load the ``ou`` section from model_config.yaml."""
    import yaml

    with open(config.MODEL_CONFIG_FILE) as f:
        return yaml.safe_load(f)["ou"]


def run_training() -> None:
    """Execute the full ensemble training pipeline."""
    cfg = _load_config()
    version = cfg["model_version"]
    n_models = cfg["models_to_train"]
    top_n = cfg.get("top_models", 3)
    threshold = cfg.get("accuracy_threshold", 0.50)
    model_weights = cfg.get("model_weights", [0.4, 0.35, 0.25])
    starting_capital = cfg.get("starting_capital", 100.0)
    train_cfg = cfg.get("train", {})
    test_size = train_cfg.get("test_size", 0.2)
    hyperparams = cfg.get("hyperparameters")

    out_dir = config.MODELS_DIR / version / "algorithm"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_prepare(min_training_seasons=2)
    all_consensus_picks: list[pd.DataFrame] = []

    # Outer loop: dates
    for date_index, current_date in enumerate(data.test_dates):
        logger.info("Processing date %s (%d/%d)",
                     current_date, date_index + 1, len(data.test_dates))

        # Inner loop: train N models for this date
        models = train_ensemble_for_date(
            data.df,
            current_date,
            date_index,
            n_models=n_models,
            test_size=test_size,
            accuracy_threshold=threshold,
            hyperparameters=hyperparams,
        )
        if not models:
            logger.warning("No models passed threshold for %s.", current_date)
            continue

        # Season-weighted model selection
        w_cur, w_last = compute_season_progress(current_date)
        scored = compute_weighted_accuracy(models, w_cur, w_last)
        top = select_top_models(scored, top_n=top_n)

        if len(top) < top_n:
            logger.warning(
                "Only %d models available (need %d) for %s. Skipping.",
                len(top),
                top_n,
                current_date,
            )
            continue

        # Consensus prediction
        test_df = data.df[data.df[DATE_COLUMN] == current_date]
        consensus = predict_with_consensus(
            top, test_df, w_cur, w_last, model_weights
        )
        if consensus is None:
            logger.info("No consensus for %s.", current_date)
            continue

        all_consensus_picks.append(consensus)

    if not all_consensus_picks:
        logger.error("No consensus picks across all dates. Aborting.")
        return

    all_picks_df = pd.concat(all_consensus_picks, ignore_index=True)
    all_picks_df["Correct Prediction"] = (
        all_picks_df["Actual"] == all_picks_df["Predicted"]
    ).astype(int)

    # Save combined picks
    all_picks_df.to_csv(out_dir / "combined_picks.csv", index=False)
    logger.info("Saved %d consensus picks.", len(all_picks_df))

    # --- Accuracy by confidence bin ---
    acc_by_conf = (
        all_picks_df.groupby("Confidence Bin", observed=False)
        .agg(
            Average_Adjusted_Score=("Final Algorithm Score", "mean"),
            Actual_Accuracy=("Correct Prediction", "mean"),
            Prediction_Count=("Correct Prediction", "count"),
        )
        .reset_index()
    )
    acc_by_conf.to_csv(out_dir / "accuracy_by_confidence_overall.csv", index=False)
    plot_accuracy_by_confidence(
        acc_by_conf,
        out_dir / "estimated_vs_actual_accuracy_by_confidence.png",
    )

    # --- Accuracy by algorithm score bin ---
    algo_bins = np.arange(0.0, 1.05, 0.05)
    algo_labels = [f"{int(b * 100)}-{int((b + 0.05) * 100)}%" for b in algo_bins[:-1]]
    all_picks_df["Algorithm Score Bin"] = pd.cut(
        all_picks_df["Final Algorithm Score"],
        bins=algo_bins,
        labels=algo_labels,
        include_lowest=True,
    )

    algo_score_acc = (
        all_picks_df.groupby("Algorithm Score Bin", observed=False)
        .agg(
            Prediction_Count=("Correct Prediction", "count"),
            Actual_Accuracy=("Correct Prediction", "mean"),
        )
        .reset_index()
    )
    algo_score_acc.to_csv(out_dir / "accuracy_by_algorithm_score_overall.csv", index=False)
    plot_accuracy_by_algorithm_score(
        algo_score_acc,
        out_dir / "accuracy_by_algorithm_score.png",
    )

    # --- Accuracy by algorithm score + season ---
    algo_score_season = (
        all_picks_df.groupby(["Algorithm Score Bin", "Season"], observed=False)
        .agg(
            Prediction_Count=("Correct Prediction", "count"),
            Actual_Accuracy=("Correct Prediction", "mean"),
        )
        .reset_index()
    )
    algo_score_season.to_csv(out_dir / "accuracy_by_algorithm_score_season.csv", index=False)
    plot_accuracy_by_algorithm_score_season(
        algo_score_season,
        out_dir / "accuracy_by_algorithm_score_season.png",
    )

    # --- Financial simulation ---
    high_acc_bins = algo_score_acc[
        algo_score_acc["Actual_Accuracy"] > 0.525
    ]["Algorithm Score Bin"].tolist()

    if high_acc_bins:
        selected = all_picks_df[
            all_picks_df["Algorithm Score Bin"].isin(high_acc_bins)
        ].copy()

        selected = simulate_betting(selected, starting_capital=starting_capital)
        selected.to_csv(out_dir / "simulation_picks.csv", index=False)

        plot_cumulative_profit(selected, out_dir / "cumulative_profit.png")

        stats = compute_performance_stats(selected)
        write_performance_stats(stats, out_dir / "performance_statistics.txt")

        logger.info("High-accuracy bins used for simulation: %s", high_acc_bins)
    else:
        logger.warning("No algorithm score bins with accuracy > 52.5%%. Skipping simulation.")

    logger.info("Training complete. Results saved to %s", out_dir)
