import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PFF_DATA_DIR = DATA_DIR / "pff"
PFR_DATA_DIR = DATA_DIR / "pfr"
OVERUNDER_DATA_DIR = DATA_DIR / "over-under"

SEASONS = os.environ.get("NFL_SEASONS", "2024").split(",")
START_YEAR = int(os.environ.get("NFL_START_YEAR", "2024"))
END_YEAR = int(os.environ.get("NFL_END_YEAR", "2024"))
MAX_WEEK = int(os.environ.get("NFL_MAX_WEEK", "18"))

# All file paths used by the pipeline
PFR_BOXSCORES_FILE = PFR_DATA_DIR / "boxscores_urls.txt"
PFR_GAME_DATA_FILE = PFR_DATA_DIR / "regular_game_data.csv"
PFR_NORMALIZED_FILE = PFR_DATA_DIR / "normalized_pfr_odds.csv"
PFR_FINAL_FILE = PFR_DATA_DIR / "final_pfr_odds.csv"
PFF_RAW_FILE = PFF_DATA_DIR / "raw_team_data.csv"
PFF_DATES_FILE = PFF_DATA_DIR / "dates_team_data.csv"
PFF_NORMALIZED_FILE = PFF_DATA_DIR / "normalized_team_data.csv"
MERGED_FILE = DATA_DIR / "pff_and_pfr_data.csv"
OVERUNDER_RAW = OVERUNDER_DATA_DIR / "raw-dataset.csv"
OVERUNDER_AVERAGES = OVERUNDER_DATA_DIR / "data-w-averages.csv"
OVERUNDER_GP = OVERUNDER_DATA_DIR / "v1-dataset-gp.csv"
OVERUNDER_RANKED = OVERUNDER_DATA_DIR / "v1-dataset-gp-ranked.csv"
PROXY_FILE = PROJECT_ROOT / "proxies" / "proxies.csv"
