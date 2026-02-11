"""Full pipeline orchestrator for NFL data collection and processing."""

import logging

from sports_quant import _config as config
from sports_quant.scrapers.pff import scrape_pff_data
from sports_quant.scrapers.pfr_urls import collect_boxscore_urls
from sports_quant.scrapers.pfr import scrape_all_game_info
from sports_quant.parsers.pff_dates import extract_dates
from sports_quant.parsers.pff_teams import normalize_pff_teams
from sports_quant.parsers.pfr_dates import normalize_pfr_dates
from sports_quant.parsers.pfr_teams import extract_pfr_teams
from sports_quant.processing.merge import merge_datasets
from sports_quant.processing.over_under import process_over_under
from sports_quant.processing.rolling_averages import compute_rolling_averages
from sports_quant.processing.games_played import add_games_played
from sports_quant.processing.rankings import compute_rankings

logger = logging.getLogger(__name__)


def ensure_dirs() -> None:
    """Create data output directories if they don't exist."""
    for d in (config.PFF_DATA_DIR, config.PFR_DATA_DIR, config.OVERUNDER_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run_pff_pipeline() -> None:
    """Run the full PFF scraping and parsing pipeline."""
    ensure_dirs()
    logger.info("Scraping PFF data...")
    scrape_pff_data()
    logger.info("Extracting PFF dates...")
    extract_dates()
    logger.info("Normalizing PFF team names...")
    normalize_pff_teams()
    logger.info("PFF pipeline complete.")


def run_pfr_pipeline() -> None:
    """Run the full PFR scraping and parsing pipeline."""
    ensure_dirs()
    logger.info("Collecting PFR boxscore URLs...")
    collect_boxscore_urls()
    logger.info("Scraping PFR game data...")
    scrape_all_game_info()
    logger.info("Normalizing PFR dates...")
    normalize_pfr_dates()
    logger.info("Extracting PFR team names...")
    extract_pfr_teams()
    logger.info("PFR pipeline complete.")


def run_processing_pipeline() -> None:
    """Run the full post-processing pipeline (assumes PFF + PFR data exist)."""
    logger.info("Merging datasets...")
    merge_datasets()
    logger.info("Processing over/under...")
    process_over_under()
    logger.info("Computing rolling averages...")
    compute_rolling_averages()
    logger.info("Adding games played...")
    add_games_played()
    logger.info("Computing rankings...")
    compute_rankings()
    logger.info("Processing pipeline complete.")


def run_full_pipeline() -> None:
    """Run the entire pipeline end-to-end."""
    run_pff_pipeline()
    run_pfr_pipeline()
    run_processing_pipeline()
    logger.info("Full pipeline complete.")
