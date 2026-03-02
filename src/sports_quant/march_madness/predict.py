"""March Madness prediction generator.

Loads trained models and generates both standard and debiased predictions
for a target year.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sports_quant import _config as config
from sports_quant.march_madness import _config as mm_config
from sports_quant.march_madness._data import load_prediction_data
from sports_quant.march_madness._debiasing import swap_team_columns

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load the 'march_madness' section from model_config.yaml."""
    with open(config.MODEL_CONFIG_FILE) as f:
        full = yaml.safe_load(f)
    return full["march_madness"]


def run_prediction(
    model_version: str | None = None,
    model_rank: int = 1,
) -> None:
    """Generate predictions for the current year using a trained model.

    Produces both standard and debiased prediction CSVs.

    Args:
        model_version: Model version to load. Defaults to config value.
        model_rank: Which top model to use (1, 2, or 3). Defaults to 1.
    """
    cfg = _load_config()
    model_version = model_version or cfg["model_version"]

    # Load model
    model_path = (
        mm_config.MM_MODELS_DIR / model_version / f"top_model_{model_rank}.joblib"
    )
    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)

    # Load prediction data
    X_pred, team_info = load_prediction_data()

    # Standard predictions
    y_pred_proba = model.predict_proba(X_pred)[:, 1]
    y_pred = model.predict(X_pred)

    results = pd.DataFrame({
        "Year": team_info["YEAR"],
        "Round": team_info["CURRENT ROUND"],
        "Team1": team_info["Team1"],
        "Seed1": team_info["Seed1"],
        "Team2": team_info["Team2"],
        "Seed2": team_info["Seed2"],
        "Prediction": y_pred,
        "Team1_Win_Probability": y_pred_proba,
        "Team2_Win_Probability": 1 - y_pred_proba,
        "Predicted_Winner": team_info.apply(
            lambda row: row["Team1"] if y_pred[row.name] == 1 else row["Team2"],
            axis=1,
        ),
    })

    # Debiased predictions
    X_swapped = swap_team_columns(X_pred)
    swapped_proba = model.predict_proba(X_swapped)[:, 1]
    swapped_team2_win_proba = 1 - swapped_proba
    debiased_proba = (y_pred_proba + swapped_team2_win_proba) / 2
    debiased_pred = [1 if p >= 0.5 else 0 for p in debiased_proba]

    debiased_results = pd.DataFrame({
        "Year": team_info["YEAR"],
        "Round": team_info["CURRENT ROUND"],
        "Team1": team_info["Team1"],
        "Seed1": team_info["Seed1"],
        "Team2": team_info["Team2"],
        "Seed2": team_info["Seed2"],
        "Prediction": debiased_pred,
        "Team1_Win_Probability": debiased_proba,
        "Team2_Win_Probability": 1 - np.array(debiased_proba),
        "Predicted_Winner": team_info.apply(
            lambda row: (
                row["Team1"] if debiased_pred[row.name] == 1 else row["Team2"]
            ),
            axis=1,
        ),
    })

    # Save results
    out_dir = mm_config.MM_PREDICTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    year = team_info["YEAR"].iloc[0] if len(team_info) > 0 else "unknown"
    standard_path = out_dir / f"march_madness_{year}_predictions.csv"
    debiased_path = out_dir / f"march_madness_{year}_debiased_predictions.csv"

    results.to_csv(standard_path, index=False)
    debiased_results.to_csv(debiased_path, index=False)

    logger.info("Standard predictions saved to %s", standard_path)
    logger.info("Debiased predictions saved to %s", debiased_path)


if __name__ == "__main__":
    run_prediction()
