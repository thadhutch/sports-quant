"""Click CLI entry point for sports-quant."""

import click

from sports_quant import __version__


@click.group()
@click.version_option(version=__version__, prog_name="sports-quant")
def cli():
    """Sports analytics pipeline for NFL and March Madness prediction."""


@cli.group()
def scrape():
    """Scrape data from external sources."""


@scrape.command()
def pff():
    """Run the full PFF scraping and parsing pipeline."""
    from sports_quant.pipeline import run_pff_pipeline
    run_pff_pipeline()


@scrape.command()
def pfr():
    """Run the full PFR scraping and parsing pipeline."""
    from sports_quant.pipeline import run_pfr_pipeline
    run_pfr_pipeline()


@cli.group()
def process():
    """Run post-processing steps."""


@process.command()
def merge():
    """Merge PFF and PFR datasets."""
    from sports_quant.processing.merge import merge_datasets
    merge_datasets()


@process.command(name="over-under")
def over_under():
    """Extract over/under betting information."""
    from sports_quant.processing.over_under import process_over_under
    process_over_under()


@process.command()
def averages():
    """Compute rolling averages for PFF statistics."""
    from sports_quant.processing.rolling_averages import compute_rolling_averages
    compute_rolling_averages()


@process.command(name="games-played")
def games_played():
    """Add cumulative games played columns."""
    from sports_quant.processing.games_played import add_games_played
    add_games_played()


@process.command()
def rankings():
    """Compute feature rankings."""
    from sports_quant.processing.rankings import compute_rankings
    compute_rankings()


@process.command()
def all():
    """Run the full post-processing pipeline."""
    from sports_quant.pipeline import run_processing_pipeline
    run_processing_pipeline()


@cli.group()
def chart():
    """Generate data visualizations."""


@chart.command(name="upset-rate")
def upset_rate():
    """Generate upset rate by spread size chart."""
    from sports_quant.visualizations.upset_rate import generate_upset_rate_chart
    generate_upset_rate_chart()


@chart.command(name="underdog-teams")
def underdog_teams():
    """Generate best underdog teams chart."""
    from sports_quant.visualizations.underdog_teams import generate_best_underdog_teams_chart
    generate_best_underdog_teams_chart()


@chart.command(name="dogs-that-bite")
def dogs_that_bite():
    """Generate dogs that bite (7+ point underdogs) chart."""
    from sports_quant.visualizations.underdog_teams import generate_dogs_that_bite_chart
    generate_dogs_that_bite_chart()


@chart.command(name="correlation-heatmap")
def correlation_heatmap():
    """Generate PFF grade correlation heatmap with scoring & O/U outcomes."""
    from sports_quant.visualizations.correlation_heatmap import generate_correlation_heatmap
    generate_correlation_heatmap()


@chart.command(name="pff-grade-vs-points")
def pff_grade_vs_points():
    """Generate PFF grade vs total points small multiples scatter plot."""
    from sports_quant.visualizations.pff_grade_vs_points import generate_pff_grade_vs_points
    generate_pff_grade_vs_points()


@chart.command(name="feature-importance")
def feature_importance():
    """Generate feature importance for O/U prediction chart."""
    from sports_quant.visualizations.feature_importance import generate_feature_importance
    generate_feature_importance()


@chart.command(name="pff-vs-vegas-spread")
def pff_vs_vegas_spread():
    """Generate PFF grade differential vs Vegas spread scatter plot."""
    from sports_quant.visualizations.pff_vs_vegas_spread import generate_pff_vs_vegas_spread
    generate_pff_vs_vegas_spread()


@chart.command(name="ou-line-vs-actual")
def ou_line_vs_actual():
    """Generate O/U line vs actual total score scatter plot."""
    from sports_quant.visualizations.ou_line_vs_actual import generate_ou_line_vs_actual
    generate_ou_line_vs_actual()


@chart.command(name="team-ranking-heatmap")
def team_ranking_heatmap():
    """Generate team ranking heatmap across PFF grade categories."""
    from sports_quant.visualizations.team_ranking_heatmap import generate_team_ranking_heatmap
    generate_team_ranking_heatmap()


@chart.command(name="vegas-line-accuracy")
def vegas_line_accuracy():
    """Generate Vegas spread accuracy histogram."""
    from sports_quant.visualizations.vegas_line_accuracy import generate_vegas_line_accuracy
    generate_vegas_line_accuracy()


@chart.command(name="vegas-accuracy-conditions")
def vegas_accuracy_conditions():
    """Generate Vegas spread accuracy box plots by surface/roof type."""
    from sports_quant.visualizations.vegas_accuracy_by_conditions import generate_vegas_accuracy_by_conditions
    generate_vegas_accuracy_by_conditions()


@chart.command(name="team-trajectory")
def team_trajectory():
    """Generate team performance trajectory line chart."""
    from sports_quant.visualizations.team_performance_trajectory import generate_team_performance_trajectory
    generate_team_performance_trajectory()


@chart.command(name="ou-accuracy-by-range")
def ou_accuracy_by_range():
    """Generate O/U accuracy by line range bar chart."""
    from sports_quant.visualizations.ou_accuracy_by_line_range import generate_ou_accuracy_by_line_range
    generate_ou_accuracy_by_line_range()


@chart.command(name="off-vs-def-correlation")
def off_vs_def_correlation():
    """Generate offensive vs defensive PFF grade correlation scatter plot."""
    from sports_quant.visualizations.off_vs_def_correlation import generate_off_vs_def_correlation
    generate_off_vs_def_correlation()


@chart.command(name="team-radar")
def team_radar():
    """Generate team PFF grade radar/spider chart."""
    from sports_quant.visualizations.team_radar_chart import generate_team_radar_chart
    generate_team_radar_chart()


@chart.command(name="wins-vs-pff-grade")
def wins_vs_pff_grade():
    """Generate win percentage vs composite PFF grade scatter plot."""
    from sports_quant.visualizations.wins_vs_pff_grade import generate_wins_vs_pff_grade
    generate_wins_vs_pff_grade()


@chart.command(name="grade-differential-upsets")
def grade_differential_upsets():
    """Generate biggest PFF grade-differential upsets bar chart."""
    from sports_quant.visualizations.grade_differential_upsets import generate_grade_differential_upsets
    generate_grade_differential_upsets()


@chart.command(name="early-vs-late-grades")
def early_vs_late_grades():
    """Generate early vs late season PFF grade comparison chart."""
    from sports_quant.visualizations.early_vs_late_grades import generate_early_vs_late_grades
    generate_early_vs_late_grades()


@chart.command(name="grade-stability")
def grade_stability():
    """Generate PFF grade stability by games played line chart."""
    from sports_quant.visualizations.grade_stability import generate_grade_stability
    generate_grade_stability()


@cli.group()
def model():
    """Train and evaluate O/U prediction models."""


@model.command()
def train():
    """Train the ensemble O/U prediction model."""
    from sports_quant.modeling.train import run_training
    run_training()


@model.command()
def backtest():
    """Run walk-forward backtesting on the O/U model."""
    from sports_quant.modeling.backtest import run_backtest
    run_backtest()


@model.command()
def analyze():
    """Analyze what distinguishes reliable picks from unreliable ones."""
    from sports_quant.modeling.analysis import run_analysis
    run_analysis()


@cli.command()
def pipeline():
    """Run the entire pipeline end-to-end."""
    from sports_quant.pipeline import run_full_pipeline
    run_full_pipeline()


# -- March Madness commands ---------------------------------------------------


@cli.group(name="march-madness")
def march_madness():
    """March Madness NCAA basketball prediction pipeline."""


@march_madness.command()
def scrape_kenpom():
    """Scrape KenPom ratings from web archive."""
    from sports_quant.march_madness.scrapers.kenpom import scrape_kenpom
    scrape_kenpom()


@march_madness.command(name="clean-kenpom")
def clean_kenpom():
    """Run the KenPom data cleaning pipeline."""
    from sports_quant.march_madness.processing.clean_kenpom import run_kenpom_cleaning_pipeline
    run_kenpom_cleaning_pipeline()


@march_madness.command()
def preprocess():
    """Preprocess matchup data (pair rows into Team1 vs Team2)."""
    from sports_quant.march_madness.processing.preprocess_matchups import preprocess_matchups
    preprocess_matchups()


@march_madness.command(name="merge-stats")
def merge_stats():
    """Merge matchups with KenPom statistics."""
    from sports_quant.march_madness.processing.merge_matchups_stats import merge_matchups_stats
    merge_matchups_stats()


@march_madness.command()
def train():
    """Train LightGBM ensemble for March Madness prediction."""
    from sports_quant.march_madness.train import run_training
    run_training()


@march_madness.command()
def backtest():
    """Run sequential year-by-year backtesting."""
    from sports_quant.march_madness.backtest import run_backtest
    run_backtest()


@march_madness.command()
def predict():
    """Generate predictions for the current year."""
    from sports_quant.march_madness.predict import run_prediction
    run_prediction()


@march_madness.command()
@click.option(
    "--year", "-y",
    multiple=True,
    type=int,
    help="Tournament year(s) to render. Repeatable. Default: all backtest years.",
)
@click.option(
    "--source", "-s",
    multiple=True,
    type=click.Choice(["ensemble", "debiased"], case_sensitive=False),
    help="Prediction source(s) to render. Default: both.",
)
@click.option(
    "--version", "-v",
    default="v1",
    help="Model version directory (default: v1).",
)
def bracket(year, source, version):
    """Render bracket visualisations from backtest results."""
    from sports_quant.march_madness._bracket_cli import run_bracket_visualisation

    years = list(year) if year else None
    sources = list(source) if source else None

    outputs = run_bracket_visualisation(
        years=years,
        sources=sources,
        version=version,
    )

    click.echo(f"Rendered {len(outputs)} bracket files:")
    for path in outputs:
        click.echo(f"  {path}")
