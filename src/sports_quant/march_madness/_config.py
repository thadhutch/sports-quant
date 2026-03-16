"""March Madness path configuration."""

from sports_quant import _config as config

MM_DATA_DIR = config.MM_DATA_DIR
MM_KENPOM_DIR = MM_DATA_DIR / "kenpom"
MM_MATCHUPS_DIR = MM_DATA_DIR / "matchups"
MM_MODELS_DIR = MM_DATA_DIR / "models"
MM_BACKTEST_DIR = MM_DATA_DIR / "backtest"
MM_PREDICTIONS_DIR = MM_DATA_DIR / "predictions"
MM_PLOTS_DIR = MM_DATA_DIR / "plots"
MM_CLEANING_DIR = MM_KENPOM_DIR / "cleaning_steps"

# Key files
MM_KENPOM_DATA = MM_KENPOM_DIR / "kenpom_data.csv"
MM_KENPOM_RAW = MM_KENPOM_DIR / "kenpom_data.csv"
MM_KENPOM_HISTORICAL = MM_KENPOM_DIR / "kenpom_2017_2024.csv"
MM_MATCHUPS_RAW = MM_MATCHUPS_DIR / "matchups.csv"
MM_MATCHUPS_RESTRUCTURED = MM_MATCHUPS_DIR / "restructured_matchups.csv"
MM_TRAINING_DATA = MM_DATA_DIR / "training_data.csv"
MM_PREDICTION_DATA = MM_DATA_DIR / "prediction_data.csv"
