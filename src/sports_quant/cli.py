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


@march_madness.command(name="save-version")
@click.option(
    "--version", "-v",
    default=None,
    help="Version identifier (default: from model_config.yaml).",
)
@click.option(
    "--description", "-d",
    required=True,
    help="Short description of what changed in this version.",
)
def save_version(version, description):
    """Snapshot current backtest results as a named version with metrics."""
    import yaml

    from sports_quant._config import MODEL_CONFIG_FILE
    from sports_quant.march_madness._versioning import save_version as _save

    if version is None:
        cfg = yaml.safe_load(MODEL_CONFIG_FILE.read_text())
        version = cfg["march_madness"]["model_version"]

    vm = _save(version=version, description=description)

    click.echo(f"Version {vm.version} saved ({vm.created_at})")
    click.echo(f"  Description: {vm.description}")
    click.echo(f"  Bracket accuracy (avg): {vm.avg_bracket_accuracy:.1%}")
    click.echo(f"  Survivor rounds (avg):  {vm.avg_survivor_rounds:.1f}")

    click.echo("\nBracket metrics:")
    for m in vm.bracket_metrics:
        click.echo(
            f"  {m.year} {m.source:>10s}: "
            f"{m.overall_accuracy:.1%} "
            f"({m.correct_games}/{m.total_games})"
        )

    click.echo("\nSurvivor metrics:")
    for m in vm.survivor_metrics:
        status = "SURVIVED" if m.survived_all else f"out R{m.rounds_survived + 1}"
        click.echo(
            f"  {m.year} {m.strategy:>8s}: "
            f"{m.rounds_survived}/6 rounds ({status})"
        )


@march_madness.command(name="list-versions")
def list_versions_cmd():
    """List all saved model versions."""
    from sports_quant.march_madness._versioning import list_versions

    registry = list_versions()

    if not registry:
        click.echo("No saved versions. Run save-version first.")
        return

    # Header
    click.echo(
        f"{'Version':<10} {'Accuracy':>10} {'Survivor':>10} "
        f"{'Created':>22}  Description"
    )
    click.echo("-" * 80)

    for entry in registry:
        acc = entry.get("avg_bracket_accuracy", 0)
        surv = entry.get("avg_survivor_rounds", 0)
        click.echo(
            f"{entry['version']:<10} "
            f"{acc:>9.1%} "
            f"{surv:>8.1f}R "
            f"{entry['created_at'][:19]:>22}  "
            f"{entry.get('description', '')}"
        )


@march_madness.command(name="compare-versions")
@click.argument("version_a")
@click.argument("version_b")
def compare_versions_cmd(version_a, version_b):
    """Compare two saved versions side by side.

    VERSION_A is the baseline, VERSION_B is the candidate.
    """
    from sports_quant.march_madness._versioning import compare_versions

    result = compare_versions(version_a, version_b)

    # Bracket comparison
    click.echo(f"\nBracket Accuracy: {version_a} vs {version_b}")
    click.echo("-" * 65)
    click.echo(
        f"{'Year':<6} {'Source':<12} "
        f"{version_a:>10} {version_b:>10} {'Delta':>10}"
    )

    for row in result["bracket"]:
        acc_a = f"{row[f'{version_a}_accuracy']:.1%}" if row[f"{version_a}_accuracy"] is not None else "  —"
        acc_b = f"{row[f'{version_b}_accuracy']:.1%}" if row[f"{version_b}_accuracy"] is not None else "  —"
        delta = ""
        if row["delta"] is not None:
            sign = "+" if row["delta"] >= 0 else ""
            delta = f"{sign}{row['delta']:.1%}"
        click.echo(
            f"{row['year']:<6} {row['source']:<12} "
            f"{acc_a:>10} {acc_b:>10} {delta:>10}"
        )

    # Survivor comparison
    click.echo(f"\nSurvivor Rounds: {version_a} vs {version_b}")
    click.echo("-" * 65)
    click.echo(
        f"{'Year':<6} {'Strategy':<12} "
        f"{version_a:>10} {version_b:>10} {'Delta':>10}"
    )

    for row in result["survivor"]:
        ra = f"{row[f'{version_a}_rounds']}/6" if row[f"{version_a}_rounds"] is not None else "—"
        rb = f"{row[f'{version_b}_rounds']}/6" if row[f"{version_b}_rounds"] is not None else "—"
        delta = ""
        if row["delta"] is not None:
            sign = "+" if row["delta"] >= 0 else ""
            delta = f"{sign}{row['delta']}"
        click.echo(
            f"{row['year']:<6} {row['strategy']:<12} "
            f"{ra:>10} {rb:>10} {delta:>10}"
        )

    # Summary
    s = result["summary"]
    click.echo(f"\nSummary:")
    acc_delta = s["accuracy_delta"]
    acc_sign = "+" if acc_delta >= 0 else ""
    click.echo(
        f"  Bracket accuracy: {s[f'{version_a}_avg_accuracy']:.1%} → "
        f"{s[f'{version_b}_avg_accuracy']:.1%} "
        f"({acc_sign}{acc_delta:.1%})"
    )
    surv_delta = s["survivor_delta"]
    surv_sign = "+" if surv_delta >= 0 else ""
    click.echo(
        f"  Survivor rounds:  {s[f'{version_a}_avg_survivor_rounds']:.1f} → "
        f"{s[f'{version_b}_avg_survivor_rounds']:.1f} "
        f"({surv_sign}{surv_delta:.1f})"
    )
