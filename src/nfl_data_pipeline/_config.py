import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", "data"))
PFF_DATA_DIR = DATA_DIR / "pff"
PFR_DATA_DIR = DATA_DIR / "pfr"
OVERUNDER_DATA_DIR = DATA_DIR / "over-under"

PROXY_FILE = Path(os.environ.get("NFL_PROXY_FILE", "proxies/proxies.csv"))

SEASONS = os.environ.get("NFL_SEASONS", "2025").split(",")
START_YEAR = int(os.environ.get("NFL_START_YEAR", "2025"))
END_YEAR = int(os.environ.get("NFL_END_YEAR", "2025"))
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

CHARTS_DIR = DATA_DIR / "charts"
LOGOS_DIR = DATA_DIR / "logos"
UPSET_RATE_CHART = CHARTS_DIR / "upset_rate_by_spread.png"
BEST_UNDERDOG_TEAMS_CHART = CHARTS_DIR / "best_underdog_teams.png"
DOGS_THAT_BITE_CHART = CHARTS_DIR / "dogs_that_bite.png"
CORRELATION_HEATMAP_CHART = CHARTS_DIR / "correlation_heatmap.png"
PFF_GRADE_VS_POINTS_CHART = CHARTS_DIR / "pff_grade_vs_points.png"
FEATURE_IMPORTANCE_CHART = CHARTS_DIR / "feature_importance.png"
PFF_VS_VEGAS_SPREAD_CHART = CHARTS_DIR / "pff_vs_vegas_spread.png"
OU_LINE_VS_ACTUAL_CHART = CHARTS_DIR / "ou_line_vs_actual.png"
TEAM_RANKING_HEATMAP_CHART = CHARTS_DIR / "team_ranking_heatmap.png"
VEGAS_LINE_ACCURACY_CHART = CHARTS_DIR / "vegas_line_accuracy.png"
VEGAS_ACCURACY_CONDITIONS_CHART = CHARTS_DIR / "vegas_accuracy_by_conditions.png"
TEAM_TRAJECTORY_CHART = CHARTS_DIR / "team_performance_trajectory.png"
OU_ACCURACY_BY_RANGE_CHART = CHARTS_DIR / "ou_accuracy_by_line_range.png"
OFF_VS_DEF_CORRELATION_CHART = CHARTS_DIR / "off_vs_def_correlation.png"
