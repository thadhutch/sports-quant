"""Bracket visualisation orchestration for CLI integration.

Loads backtest results, builds brackets, and renders all outputs
(SVG brackets, comparison brackets, accuracy charts) for one or
more tournament years.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sports_quant.march_madness._bracket import Bracket
from sports_quant.march_madness._bracket_builder import (
    build_actual_bracket,
    build_predicted_bracket,
    compare_brackets,
)
from sports_quant.march_madness._config import (
    MM_BACKTEST_DIR,
    MM_MATCHUPS_RESTRUCTURED,
    load_models,
)
from sports_quant.march_madness.bracket_plots import (
    render_accuracy,
    render_accuracy_comparison,
    render_bracket,
    render_bracket_with_survivor,
    render_comparison,
    render_survivor_journey,
    render_survivor_multi_year,
)

logger = logging.getLogger(__name__)


def _run_simulation_for_year(
    *,
    year: int,
    version_dir: Path,
    matchups_df: pd.DataFrame,
    feature_lookup,
) -> Bracket | None:
    """Run forward bracket simulation for a single year.

    Loads trained models and runs deterministic simulation from R64→NCG.
    Returns the evaluated bracket or None on failure.
    """
    from sports_quant.march_madness.simulate import simulate_bracket_deterministic

    models_dir = version_dir / str(year) / "models"
    if not models_dir.is_dir():
        logger.warning("Models dir not found for %d: %s", year, models_dir)
        return None

    models = load_models(models_dir)
    if not models:
        logger.warning("No models found for %d", year)
        return None

    actual = build_actual_bracket(matchups_df, year)

    try:
        result = simulate_bracket_deterministic(
            year=year,
            models=models,
            feature_lookup=feature_lookup,
            matchups_df=matchups_df,
            actual_bracket=actual,
        )
        return result.bracket
    except (KeyError, ValueError) as exc:
        logger.warning("Simulation failed for %d: %s", year, exc)
        return None


# Sources that map to specific backtest CSV filenames and column conventions
_SOURCE_CSV: dict[str, str] = {
    "ensemble": "ensemble_results.csv",
    "debiased": "debiased_results.csv",
}

_DEFAULT_SOURCES = ("ensemble", "debiased")


def resolve_backtest_paths(
    *,
    backtest_dir: Path,
    version: str,
    year: int,
) -> dict[str, Any]:
    """Resolve and validate file paths for a single backtest year.

    Args:
        backtest_dir: Root backtest directory (e.g. ``data/march-madness/backtest``).
        version: Model version (e.g. ``"v1"``).
        year: Tournament year.

    Returns:
        Dictionary with keys ``year``, ``ensemble``, ``debiased``, ``output_dir``.

    Raises:
        FileNotFoundError: If the year directory or required CSVs are missing.
    """
    year_dir = backtest_dir / version / str(year)
    if not year_dir.is_dir():
        msg = f"Backtest directory not found for {year}: {year_dir}"
        raise FileNotFoundError(msg)

    ensemble_path = year_dir / "ensemble_results.csv"
    if not ensemble_path.exists():
        msg = f"ensemble_results.csv not found in {year_dir}"
        raise FileNotFoundError(msg)

    debiased_path = year_dir / "debiased_results.csv"
    output_dir = year_dir / "plots" / "brackets"

    return {
        "year": year,
        "ensemble": ensemble_path,
        "debiased": debiased_path if debiased_path.exists() else None,
        "output_dir": output_dir,
    }


def render_year_brackets(
    *,
    year: int,
    matchups_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    debiased_df: pd.DataFrame | None,
    output_dir: Path,
    sources: tuple[str, ...] = _DEFAULT_SOURCES,
    simulation_bracket: "Bracket | None" = None,
) -> list[Path]:
    """Build and render all bracket visualisations for a single year.

    Renders:
    - Actual bracket (ground truth)
    - Predicted bracket for each source (ensemble, debiased, simulation)
    - Comparison bracket for each source vs actual

    Args:
        year: Tournament year.
        matchups_df: Full restructured matchups DataFrame.
        ensemble_df: Ensemble backtest results for this year.
        debiased_df: Debiased backtest results (None to skip).
        output_dir: Directory to write output files.
        sources: Which prediction sources to render.
        simulation_bracket: Pre-computed forward-sim bracket (None to skip).

    Returns:
        List of output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # 1. Actual bracket
    actual = build_actual_bracket(matchups_df, year)
    actual_path = render_bracket(
        actual, output_dir / f"{year}_actual.svg",
    )
    outputs.append(actual_path)
    logger.info("Rendered actual bracket: %s", actual_path)

    # 2. Predicted brackets + comparisons
    source_data: dict[str, pd.DataFrame | None] = {
        "ensemble": ensemble_df,
        "debiased": debiased_df,
    }

    for source in sources:
        if source == "simulation":
            # Handle simulation bracket separately.
            # The simulation bracket already has is_correct set via
            # position-based matching in simulate.py — do NOT call
            # compare_brackets (which matches by team names and would
            # set is_correct=None for games with wrong advancing teams).
            if simulation_bracket is None:
                logger.info("Skipping simulation for %d (not available)", year)
                continue

            sim_path = render_bracket(
                simulation_bracket, output_dir / f"{year}_simulation.svg",
            )
            outputs.append(sim_path)
            logger.info("Rendered simulation bracket: %s", sim_path)
            continue

        df = source_data.get(source)
        if df is None:
            logger.info("Skipping %s for %d (no data)", source, year)
            continue

        predicted = build_predicted_bracket(df, matchups_df, year, source=source)

        predicted_path = render_bracket(
            predicted, output_dir / f"{year}_{source}.svg",
        )
        outputs.append(predicted_path)
        logger.info("Rendered %s bracket: %s", source, predicted_path)

        comparison_path = render_comparison(
            predicted, actual, output_dir / f"{year}_{source}_comparison.svg",
        )
        outputs.append(comparison_path)
        logger.info("Rendered %s comparison: %s", source, comparison_path)

    return outputs


def render_multi_year_accuracy(
    *,
    compared_brackets: dict[int, Bracket],
    output_dirs: dict[int, Path],
    multi_year_output_dir: Path,
) -> list[Path]:
    """Render per-year accuracy charts and a multi-year comparison.

    Args:
        compared_brackets: Year → compared bracket (with is_correct flags).
        output_dirs: Year → output directory for per-year charts.
        multi_year_output_dir: Directory for the combined chart.

    Returns:
        List of output file paths.
    """
    outputs: list[Path] = []

    # Per-year accuracy charts
    for year, bracket in sorted(compared_brackets.items()):
        out_dir = output_dirs[year]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = render_accuracy(
            bracket, out_dir / f"{year}_accuracy.html",
        )
        outputs.append(path)
        logger.info("Rendered accuracy chart: %s", path)

    # Multi-year comparison
    if len(compared_brackets) > 1:
        multi_year_output_dir.mkdir(parents=True, exist_ok=True)
        brackets_list = [
            compared_brackets[y] for y in sorted(compared_brackets)
        ]
        path = render_accuracy_comparison(
            brackets_list,
            multi_year_output_dir / "multi_year_accuracy.html",
        )
        outputs.append(path)
        logger.info("Rendered multi-year accuracy: %s", path)

    return outputs


def _render_survivor_charts(
    *,
    backtest_dir: Path,
    version: str,
    years: list[int],
    matchups_df: pd.DataFrame,
    output_dirs: dict[int, Path],
    simulation_brackets: dict[int, Bracket],
) -> list[Path]:
    """Render survivor pool charts from metrics.json.

    Loads survivor metrics, renders per-year journey charts and
    bracket overlays, plus a multi-year summary if multiple years exist.

    Returns:
        List of output file paths.
    """
    from sports_quant.march_madness._results import SurvivorMetrics, load_metrics

    metrics_path = backtest_dir / version / "metrics.json"
    if not metrics_path.exists():
        logger.info("No metrics.json found at %s — skipping survivor charts", metrics_path)
        return []

    version_metrics = load_metrics(metrics_path)
    if not version_metrics.survivor_metrics:
        logger.info("No survivor metrics in %s — skipping", metrics_path)
        return []

    outputs: list[Path] = []
    all_survivor: list[SurvivorMetrics] = []

    for year in years:
        year_survivors = [
            m for m in version_metrics.survivor_metrics if m.year == year
        ]
        if not year_survivors:
            continue

        all_survivor.extend(year_survivors)
        output_dir = output_dirs.get(year)
        if output_dir is None:
            output_dir = backtest_dir / version / str(year) / "plots" / "brackets"

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Journey chart (Plotly HTML)
        journey_path = render_survivor_journey(
            year_survivors,
            output_dir / f"{year}_survivor_journey.html",
            year,
        )
        outputs.append(journey_path)
        logger.info("Rendered survivor journey chart: %s", journey_path)

        # 2. Bracket overlays (one SVG per strategy)
        bracket = simulation_brackets.get(year)
        if bracket is None:
            # Try building from actual results as fallback
            try:
                bracket = build_actual_bracket(matchups_df, year)
            except (KeyError, ValueError):
                logger.warning("Cannot build bracket for %d — skipping overlay", year)
                continue

        for survivor in year_survivors:
            if not survivor.picks:
                continue
            svg_path = render_bracket_with_survivor(
                bracket,
                survivor.picks,
                output_dir / f"{year}_simulation_survivor_{survivor.strategy}.svg",
                strategy_label=survivor.strategy.title(),
            )
            outputs.append(svg_path)
            logger.info("Rendered survivor overlay: %s", svg_path)

    # 3. Multi-year summary
    if len(all_survivor) > 1:
        multi_dir = backtest_dir / version
        multi_dir.mkdir(parents=True, exist_ok=True)
        multi_path = render_survivor_multi_year(
            all_survivor,
            multi_dir / "multi_year_survivor.html",
        )
        outputs.append(multi_path)
        logger.info("Rendered multi-year survivor chart: %s", multi_path)

    return outputs


def run_bracket_visualisation(
    *,
    years: list[int] | None = None,
    sources: list[str] | None = None,
    version: str = "v1",
    backtest_dir: Path | None = None,
    matchups_path: Path | None = None,
) -> list[Path]:
    """Run the full bracket visualisation pipeline.

    This is the main entry point called by the CLI command.

    Args:
        years: Tournament years to render (default: all from config).
        sources: Prediction sources to render (default: ensemble + debiased).
        version: Model version directory name.
        backtest_dir: Override backtest directory (default: from config).
        matchups_path: Override matchups CSV path (default: from config).

    Returns:
        List of all output file paths.
    """
    import yaml

    from sports_quant._config import MODEL_CONFIG_FILE

    backtest_dir = backtest_dir or MM_BACKTEST_DIR
    matchups_path = matchups_path or MM_MATCHUPS_RESTRUCTURED

    # Load config for default years
    if years is None:
        config = yaml.safe_load(MODEL_CONFIG_FILE.read_text())
        years = config["march_madness"]["backtest_years"]

    source_tuple = tuple(sources) if sources else _DEFAULT_SOURCES

    # Validate matchups file
    if not matchups_path.exists():
        msg = f"Matchups file not found: {matchups_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading matchups from %s", matchups_path)
    matchups_df = pd.read_csv(matchups_path)

    # Load simulation resources lazily (only if "simulation" source requested)
    feature_lookup = None
    if "simulation" in source_tuple:
        try:
            from sports_quant.march_madness._config import MM_KENPOM_DATA
            from sports_quant.march_madness._feature_builder import FeatureLookup

            kenpom_df = pd.read_csv(MM_KENPOM_DATA)
            feature_lookup = FeatureLookup(kenpom_df)
        except (FileNotFoundError, ImportError) as exc:
            logger.warning("Cannot load simulation resources: %s", exc)

    all_outputs: list[Path] = []
    compared_brackets: dict[int, Bracket] = {}
    output_dirs: dict[int, Path] = {}

    for year in years:
        logger.info("Processing year %d", year)

        # Resolve and validate paths
        paths = resolve_backtest_paths(
            backtest_dir=backtest_dir, version=version, year=year,
        )

        # Load backtest results
        ensemble_df = pd.read_csv(paths["ensemble"])
        debiased_df = (
            pd.read_csv(paths["debiased"])
            if paths["debiased"] is not None
            else None
        )

        output_dir = paths["output_dir"]
        output_dirs[year] = output_dir

        # Run forward simulation if requested and resources are available
        simulation_bracket = None
        if "simulation" in source_tuple and feature_lookup is not None:
            simulation_bracket = _run_simulation_for_year(
                year=year,
                version_dir=backtest_dir / version,
                matchups_df=matchups_df,
                feature_lookup=feature_lookup,
            )

        # Render brackets for this year
        year_outputs = render_year_brackets(
            year=year,
            matchups_df=matchups_df,
            ensemble_df=ensemble_df,
            debiased_df=debiased_df,
            output_dir=output_dir,
            sources=source_tuple,
            simulation_bracket=simulation_bracket,
        )
        all_outputs.extend(year_outputs)

        # Build compared bracket for accuracy charts
        # Use first available source for accuracy tracking
        primary_source = source_tuple[0]
        if primary_source == "simulation" and simulation_bracket is not None:
            compared_brackets[year] = simulation_bracket
        else:
            primary_df = (
                ensemble_df if primary_source == "ensemble" else debiased_df
            )
            if primary_df is not None:
                actual = build_actual_bracket(matchups_df, year)
                predicted = build_predicted_bracket(
                    primary_df, matchups_df, year, source=primary_source,
                )
                compared = compare_brackets(predicted, actual)
                compared_brackets[year] = compared

    # Render accuracy charts
    if compared_brackets:
        accuracy_outputs = render_multi_year_accuracy(
            compared_brackets=compared_brackets,
            output_dirs=output_dirs,
            multi_year_output_dir=backtest_dir / version,
        )
        all_outputs.extend(accuracy_outputs)

    # Render survivor pool charts
    survivor_outputs = _render_survivor_charts(
        backtest_dir=backtest_dir,
        version=version,
        years=years,
        matchups_df=matchups_df,
        output_dirs=output_dirs,
        simulation_brackets={
            y: compared_brackets[y]
            for y in compared_brackets
        },
    )
    all_outputs.extend(survivor_outputs)

    logger.info("Bracket visualisation complete: %d files", len(all_outputs))
    return all_outputs
