"""Modeling subpackage for O/U prediction ensemble training and backtesting."""

from sports_quant.modeling.backtest import run_backtest
from sports_quant.modeling.train import run_training

__all__ = ["run_training", "run_backtest"]
