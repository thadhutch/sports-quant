"""Modeling subpackage for O/U prediction ensemble training and backtesting."""

from nfl_data_pipeline.modeling.backtest import run_backtest
from nfl_data_pipeline.modeling.train import run_training

__all__ = ["run_training", "run_backtest"]
