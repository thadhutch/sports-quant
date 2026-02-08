"""nfl-data-pipeline -- NFL data pipeline combining PFF grades and PFR game data."""
__version__ = "1.0.1"

from nfl_data_pipeline.scrapers.pff import scrape_pff_data
from nfl_data_pipeline.scrapers.pfr_urls import collect_boxscore_urls
from nfl_data_pipeline.scrapers.pfr import scrape_all_game_info
from nfl_data_pipeline.processing.merge import merge_datasets
from nfl_data_pipeline.processing.over_under import process_over_under
from nfl_data_pipeline.processing.rolling_averages import compute_rolling_averages
from nfl_data_pipeline.processing.games_played import add_games_played
from nfl_data_pipeline.processing.rankings import compute_rankings
