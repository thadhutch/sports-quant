"""March Madness path configuration and shared utilities."""

import logging
from pathlib import Path

import joblib
import yaml
from lightgbm import LGBMClassifier

from sports_quant import _config as config

logger = logging.getLogger(__name__)

MM_DATA_DIR = config.MM_DATA_DIR
MM_KENPOM_DIR = MM_DATA_DIR / "kenpom"
MM_BARTTORVIK_DIR = MM_DATA_DIR / "barttorvik"
MM_MATCHUPS_DIR = MM_DATA_DIR / "matchups"
MM_SCHEDULE_DIR = MM_DATA_DIR / "schedule"
MM_MODELS_DIR = MM_DATA_DIR / "models"
MM_BACKTEST_DIR = MM_DATA_DIR / "backtest"
MM_PREDICTIONS_DIR = MM_DATA_DIR / "predictions"
MM_PLOTS_DIR = MM_DATA_DIR / "plots"
MM_CLEANING_DIR = MM_KENPOM_DIR / "cleaning_steps"

# Key files
MM_KENPOM_DATA = MM_KENPOM_DIR / "kenpom_data.csv"
MM_KENPOM_RAW = MM_KENPOM_DIR / "kenpom_data.csv"
MM_KENPOM_HISTORICAL = MM_KENPOM_DIR / "kenpom_2017_2024.csv"
MM_BARTTORVIK_DATA = MM_BARTTORVIK_DIR / "barttorvik_data.csv"
MM_MATCHUPS_RAW = MM_MATCHUPS_DIR / "matchups.csv"
MM_MATCHUPS_RESTRUCTURED = MM_MATCHUPS_DIR / "restructured_matchups.csv"
MM_SCHEDULE_DATA = MM_SCHEDULE_DIR / "tournament_schedule.csv"
MM_TRAINING_DATA = MM_DATA_DIR / "training_data.csv"
MM_PREDICTION_DATA = MM_DATA_DIR / "prediction_data.csv"


def load_mm_config() -> dict:
    """Load the 'march_madness' section from model_config.yaml.

    Raises:
        FileNotFoundError: If config file does not exist.
        KeyError: If 'march_madness' section is missing.
    """
    try:
        with open(config.MODEL_CONFIG_FILE) as f:
            full = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model config not found: {config.MODEL_CONFIG_FILE}"
        ) from None

    if "march_madness" not in full:
        raise KeyError("Missing 'march_madness' section in model config")
    return full["march_madness"]


def load_models(models_dir: Path) -> list:
    """Load all saved model files from a directory.

    Discovers all top_model_*.joblib files and loads them in order.

    Args:
        models_dir: Directory containing model .joblib files.

    Returns:
        List of loaded model objects, sorted by model number.
    """
    model_paths = sorted(
        models_dir.glob("top_model_*.joblib"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    models = [joblib.load(p) for p in model_paths]
    logger.info("Loaded %d models from %s", len(models), models_dir)
    return models


def build_lgbm(hyperparams: dict, random_seed: int) -> LGBMClassifier:
    """Construct a LGBMClassifier with all hyperparameters from config."""
    return LGBMClassifier(
        objective=hyperparams["objective"],
        metric=hyperparams["metric"],
        num_leaves=hyperparams.get("num_leaves", 31),
        max_depth=hyperparams.get("max_depth", -1),
        learning_rate=hyperparams.get("learning_rate", 0.1),
        n_estimators=hyperparams.get("n_estimators", 100),
        min_child_samples=hyperparams.get("min_child_samples", 20),
        reg_alpha=hyperparams.get("reg_alpha", 0.0),
        reg_lambda=hyperparams.get("reg_lambda", 0.0),
        subsample=hyperparams.get("subsample", 1.0),
        colsample_bytree=hyperparams.get("colsample_bytree", 1.0),
        min_split_gain=hyperparams.get("min_split_gain", 0.0),
        random_state=random_seed,
        verbose=-1,
    )
