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

MODELS_DIR = DATA_DIR / "models"
BACKTEST_DIR = DATA_DIR / "backtest"
MODEL_CONFIG_FILE = Path(os.environ.get("NFL_MODEL_CONFIG", "model_config.yaml"))

CHARTS_DIR = DATA_DIR / "charts"
LOGOS_DIR = DATA_DIR / "logos"

# line-analysis/ — Vegas line & O/U accuracy charts
LINE_ANALYSIS_CHARTS_DIR = CHARTS_DIR / "line-analysis"
UPSET_RATE_CHART = LINE_ANALYSIS_CHARTS_DIR / "upset_rate_by_spread.png"
BEST_UNDERDOG_TEAMS_CHART = LINE_ANALYSIS_CHARTS_DIR / "best_underdog_teams.png"
DOGS_THAT_BITE_CHART = LINE_ANALYSIS_CHARTS_DIR / "dogs_that_bite.png"
PFF_VS_VEGAS_SPREAD_CHART = LINE_ANALYSIS_CHARTS_DIR / "pff_vs_vegas_spread.png"
OU_LINE_VS_ACTUAL_CHART = LINE_ANALYSIS_CHARTS_DIR / "ou_line_vs_actual.png"
VEGAS_LINE_ACCURACY_CHART = LINE_ANALYSIS_CHARTS_DIR / "vegas_line_accuracy.png"
VEGAS_ACCURACY_CONDITIONS_CHART = LINE_ANALYSIS_CHARTS_DIR / "vegas_accuracy_by_conditions.png"
OU_ACCURACY_BY_RANGE_CHART = LINE_ANALYSIS_CHARTS_DIR / "ou_accuracy_by_line_range.png"

# grades/ — PFF grade exploration & correlation charts
GRADES_CHARTS_DIR = CHARTS_DIR / "grades"
CORRELATION_HEATMAP_CHART = GRADES_CHARTS_DIR / "correlation_heatmap.png"
PFF_GRADE_VS_POINTS_CHART = GRADES_CHARTS_DIR / "pff_grade_vs_points.png"
FEATURE_IMPORTANCE_CHART = GRADES_CHARTS_DIR / "feature_importance.png"
OFF_VS_DEF_CORRELATION_CHART = GRADES_CHARTS_DIR / "off_vs_def_correlation.png"
WINS_VS_PFF_GRADE_CHART = GRADES_CHARTS_DIR / "wins_vs_pff_grade.png"
GRADE_DIFFERENTIAL_UPSETS_CHART = GRADES_CHARTS_DIR / "grade_differential_upsets.png"
EARLY_VS_LATE_GRADES_CHART = GRADES_CHARTS_DIR / "early_vs_late_grades.png"
GRADE_STABILITY_CHART = GRADES_CHARTS_DIR / "grade_stability.png"

# teams/ — per-team profile & trend charts
TEAMS_CHARTS_DIR = CHARTS_DIR / "teams"
TEAM_RANKING_HEATMAP_CHART = TEAMS_CHARTS_DIR / "team_ranking_heatmap.png"
TEAM_TRAJECTORY_CHART = TEAMS_CHARTS_DIR / "team_performance_trajectory.png"
TEAM_RADAR_CHART = TEAMS_CHARTS_DIR / "team_radar.png"
