"""Walk-forward backtesting orchestrator.

Trains *N* models (outer loop), each iterating over every test date
(inner loop) using walk-forward validation: train on all data before
the current date, test on the current date.  Metrics are averaged
across all models and saved as CSVs and plots.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from sports_quant import _config as config
from sports_quant.modeling._data import load_and_prepare
from sports_quant.modeling._training import (
    CONF_BINS,
    CONF_LABELS,
    train_backtest_model_for_date,
)
from sports_quant.modeling.plots import (
    plot_accuracy_by_confidence_and_season,
    plot_accuracy_by_confidence_interval,
    plot_accuracy_by_season,
)

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load the ``ou`` section from model_config.yaml."""
    import yaml

    with open(config.MODEL_CONFIG_FILE) as f:
        return yaml.safe_load(f)["ou"]


def run_backtest() -> None:
    """Execute the full walk-forward backtesting pipeline."""
    cfg = _load_config()
    version = cfg["model_version"]
    n_models = cfg["models_to_train"]
    bt_cfg = cfg.get("backtest", {})
    min_seasons = bt_cfg.get("min_training_seasons", 2)
    test_size = bt_cfg.get("test_size", 0.8)
    hyperparams = cfg.get("hyperparameters")

    out_dir = config.BACKTEST_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_prepare(min_training_seasons=min_seasons)

    overall_accuracy_list: list[float] = []
    season_accuracy_list: list[pd.DataFrame] = []
    confidence_accuracy_list: list[pd.DataFrame] = []
    confidence_accuracy_by_season_list: list[pd.DataFrame] = []

    # Outer loop: models
    for model_idx in range(n_models):
        logger.info("Training backtest model %d / %d", model_idx + 1, n_models)

        predictions: list = []
        actuals: list = []
        dates: list = []
        seasons_list: list = []
        confidences: list = []

        # Inner loop: dates (walk-forward)
        for current_date in data.test_dates:
            result = train_backtest_model_for_date(
                data.df,
                current_date,
                model_idx,
                test_size=test_size,
                hyperparameters=hyperparams,
            )
            if result is None:
                continue

            preds, acts, dts, seas, confs = result
            predictions.extend(preds)
            actuals.extend(acts)
            dates.extend(dts)
            seasons_list.extend(seas)
            confidences.extend(confs)

        if not actuals:
            logger.warning("No predictions for backtest model %d. Skipping.", model_idx + 1)
            continue

        overall_acc = accuracy_score(actuals, predictions)
        overall_accuracy_list.append(overall_acc)
        logger.info("Backtest model %d overall accuracy: %.4f", model_idx + 1, overall_acc)

        results_df = pd.DataFrame(
            {
                "Date": dates,
                "Season": seasons_list,
                "Actual": actuals,
                "Predicted": predictions,
                "Confidence": confidences,
            }
        )

        # Accuracy by season
        season_acc = (
            results_df.groupby("Season", observed=False)
            .apply(lambda x: accuracy_score(x["Actual"], x["Predicted"]))
            .reset_index(name="Accuracy")
        )
        season_accuracy_list.append(season_acc)

        # Accuracy by confidence bin
        results_df["Confidence Bin"] = pd.cut(
            results_df["Confidence"],
            bins=CONF_BINS,
            labels=CONF_LABELS,
            include_lowest=True,
        )

        conf_acc = (
            results_df.groupby("Confidence Bin", observed=False)
            .apply(lambda x: accuracy_score(x["Actual"], x["Predicted"]))
            .reset_index(name="Accuracy")
        )
        confidence_accuracy_list.append(conf_acc)

        # Accuracy by confidence bin + season
        conf_acc_season = (
            results_df.groupby(["Confidence Bin", "Season"], observed=False)
            .apply(lambda x: accuracy_score(x["Actual"], x["Predicted"]))
            .reset_index(name="Accuracy")
        )
        confidence_accuracy_by_season_list.append(conf_acc_season)

    if not overall_accuracy_list:
        logger.error("No backtest models succeeded. Aborting.")
        return

    # Aggregate across models
    avg_overall = np.mean(overall_accuracy_list)
    logger.info("Average overall accuracy (%d models): %.4f", n_models, avg_overall)

    avg_season = pd.concat(season_accuracy_list).groupby("Season").mean().reset_index()
    avg_season["Overall Accuracy"] = avg_overall
    avg_season.to_csv(out_dir / "average_season_accuracy.csv", index=False)

    avg_conf = (
        pd.concat(confidence_accuracy_list)
        .groupby("Confidence Bin")
        .mean()
        .reset_index()
    )
    avg_conf.to_csv(out_dir / "average_confidence_accuracy.csv", index=False)

    avg_conf_season = (
        pd.concat(confidence_accuracy_by_season_list)
        .groupby(["Confidence Bin", "Season"])
        .mean()
        .reset_index()
    )
    avg_conf_season.to_csv(
        out_dir / "average_confidence_accuracy_by_season.csv", index=False
    )

    # Plots
    plot_accuracy_by_season(avg_season, n_models, out_dir / "average_accuracy_by_season.png")
    plot_accuracy_by_confidence_interval(
        avg_conf, n_models, out_dir / "average_accuracy_by_confidence.png"
    )
    plot_accuracy_by_confidence_and_season(
        avg_conf_season, n_models, out_dir / "average_accuracy_by_confidence_season.png"
    )

    logger.info("Backtest complete. Results saved to %s", out_dir)
