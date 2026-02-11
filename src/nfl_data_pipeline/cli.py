"""Click CLI entry point for nfl-data-pipeline."""

import click

from nfl_data_pipeline import __version__


@click.group()
@click.version_option(version=__version__, prog_name="nfl-pipeline")
def cli():
    """NFL data pipeline combining PFF grades and PFR game data."""


@cli.group()
def scrape():
    """Scrape data from external sources."""


@scrape.command()
def pff():
    """Run the full PFF scraping and parsing pipeline."""
    from nfl_data_pipeline.pipeline import run_pff_pipeline
    run_pff_pipeline()


@scrape.command()
def pfr():
    """Run the full PFR scraping and parsing pipeline."""
    from nfl_data_pipeline.pipeline import run_pfr_pipeline
    run_pfr_pipeline()


@cli.group()
def process():
    """Run post-processing steps."""


@process.command()
def merge():
    """Merge PFF and PFR datasets."""
    from nfl_data_pipeline.processing.merge import merge_datasets
    merge_datasets()


@process.command(name="over-under")
def over_under():
    """Extract over/under betting information."""
    from nfl_data_pipeline.processing.over_under import process_over_under
    process_over_under()


@process.command()
def averages():
    """Compute rolling averages for PFF statistics."""
    from nfl_data_pipeline.processing.rolling_averages import compute_rolling_averages
    compute_rolling_averages()


@process.command(name="games-played")
def games_played():
    """Add cumulative games played columns."""
    from nfl_data_pipeline.processing.games_played import add_games_played
    add_games_played()


@process.command()
def rankings():
    """Compute feature rankings."""
    from nfl_data_pipeline.processing.rankings import compute_rankings
    compute_rankings()


@process.command()
def all():
    """Run the full post-processing pipeline."""
    from nfl_data_pipeline.pipeline import run_processing_pipeline
    run_processing_pipeline()


@cli.group()
def chart():
    """Generate data visualizations."""


@chart.command(name="upset-rate")
def upset_rate():
    """Generate upset rate by spread size chart."""
    from nfl_data_pipeline.visualizations.upset_rate import generate_upset_rate_chart
    generate_upset_rate_chart()


@chart.command(name="underdog-teams")
def underdog_teams():
    """Generate best underdog teams chart."""
    from nfl_data_pipeline.visualizations.underdog_teams import generate_best_underdog_teams_chart
    generate_best_underdog_teams_chart()


@chart.command(name="dogs-that-bite")
def dogs_that_bite():
    """Generate dogs that bite (7+ point underdogs) chart."""
    from nfl_data_pipeline.visualizations.underdog_teams import generate_dogs_that_bite_chart
    generate_dogs_that_bite_chart()


@chart.command(name="correlation-heatmap")
def correlation_heatmap():
    """Generate PFF grade correlation heatmap with scoring & O/U outcomes."""
    from nfl_data_pipeline.visualizations.correlation_heatmap import generate_correlation_heatmap
    generate_correlation_heatmap()


@chart.command(name="pff-grade-vs-points")
def pff_grade_vs_points():
    """Generate PFF grade vs total points small multiples scatter plot."""
    from nfl_data_pipeline.visualizations.pff_grade_vs_points import generate_pff_grade_vs_points
    generate_pff_grade_vs_points()


@cli.command()
def pipeline():
    """Run the entire pipeline end-to-end."""
    from nfl_data_pipeline.pipeline import run_full_pipeline
    run_full_pipeline()
