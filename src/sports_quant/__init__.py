"""sports-quant -- NFL data pipeline combining PFF grades and PFR game data."""
__version__ = "1.1.1"

from sports_quant.scrapers.pff import scrape_pff_data
from sports_quant.scrapers.pfr_urls import collect_boxscore_urls
from sports_quant.scrapers.pfr import scrape_all_game_info
from sports_quant.processing.merge import merge_datasets
from sports_quant.processing.over_under import process_over_under
from sports_quant.processing.rolling_averages import compute_rolling_averages
from sports_quant.processing.games_played import add_games_played
from sports_quant.processing.rankings import compute_rankings
from sports_quant.modeling import run_training, run_backtest
