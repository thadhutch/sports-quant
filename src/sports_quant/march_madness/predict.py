"""March Madness prediction generator.

Loads trained models and generates both standard and debiased predictions
for a target year. Supports difference features and probability calibration.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sports_quant.march_madness._calibration import calibrate_probabilities
from sports_quant.march_madness._config import (
    MM_MODELS_DIR,
    MM_PREDICTIONS_DIR,
    load_mm_config,
    load_models,
)
from sports_quant.march_madness._data import load_prediction_data
from sports_quant.march_madness._debiasing import (
    swap_difference_features,
    swap_team_columns,
)

logger = logging.getLogger(__name__)


def run_prediction(
    model_version: str | None = None,
) -> None:
    """Generate predictions using the full ensemble of trained models.

    Loads all saved models, averages their predicted probabilities,
    then produces both standard and debiased prediction CSVs. Applies
    probability calibration if a calibrator is available.

    Args:
        model_version: Model version to load. Defaults to config value.
    """
    cfg = load_mm_config()
    model_version = model_version or cfg["model_version"]
    feature_mode = cfg.get("feature_mode", "raw")
    cal_cfg = cfg.get("calibration", {})

    # Load all models for ensemble
    models_dir = MM_MODELS_DIR / model_version
    models = load_models(models_dir)
    if not models:
        raise FileNotFoundError(f"No models found in {models_dir}")

    # Load calibrator if available
    calibrator_path = MM_MODELS_DIR / model_version / "calibrator.joblib"
    calibrator = None
    if cal_cfg.get("enabled", False) and calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)
        logger.info("Loaded calibrator from %s", calibrator_path)

    # Load prediction data
    X_pred, team_info = load_prediction_data(feature_mode=feature_mode)

    # Ensemble predictions: average probabilities across all models
    all_probas = np.array([m.predict_proba(X_pred)[:, 1] for m in models])
    y_pred_proba = np.mean(all_probas, axis=0)

    # Debiased predictions (ensemble of debiased per-model predictions)
    if feature_mode == "difference":
        X_swapped = swap_difference_features(X_pred)
    else:
        X_swapped = swap_team_columns(X_pred)
    all_swapped = np.array([m.predict_proba(X_swapped)[:, 1] for m in models])
    swapped_proba = np.mean(all_swapped, axis=0)
    debiased_proba = (y_pred_proba + (1.0 - swapped_proba)) / 2.0

    # Apply calibration
    if calibrator is not None:
        clip_min = cal_cfg.get("clip_min", 0.025)
        clip_max = cal_cfg.get("clip_max", 0.975)
        y_pred_proba = calibrate_probabilities(
            calibrator, y_pred_proba, clip_min, clip_max,
        )
        debiased_proba = calibrate_probabilities(
            calibrator, debiased_proba, clip_min, clip_max,
        )
    else:
        # Still clip even without calibrator
        y_pred_proba = np.clip(y_pred_proba, 0.025, 0.975)
        debiased_proba = np.clip(debiased_proba, 0.025, 0.975)

    y_pred = (y_pred_proba >= 0.5).astype(int)
    debiased_pred = (debiased_proba >= 0.5).astype(int)

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

    debiased_results = pd.DataFrame({
        "Year": team_info["YEAR"],
        "Round": team_info["CURRENT ROUND"],
        "Team1": team_info["Team1"],
        "Seed1": team_info["Seed1"],
        "Team2": team_info["Team2"],
        "Seed2": team_info["Seed2"],
        "Prediction": debiased_pred,
        "Team1_Win_Probability": debiased_proba,
        "Team2_Win_Probability": 1 - debiased_proba,
        "Predicted_Winner": team_info.apply(
            lambda row: (
                row["Team1"] if debiased_pred[row.name] == 1 else row["Team2"]
            ),
            axis=1,
        ),
    })

    # Save results
    out_dir = MM_PREDICTIONS_DIR
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
