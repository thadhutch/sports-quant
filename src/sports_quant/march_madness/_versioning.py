"""Model versioning for March Madness backtests.

Provides save, list, and compare operations for tracking model iterations.
Each version is a snapshot of backtest results with computed metrics,
stored alongside the backtest output directory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from sports_quant.march_madness._bracket import ROUND_ORDER
from sports_quant.march_madness._bracket_builder import (
    build_actual_bracket,
    build_predicted_bracket,
    compare_brackets,
)
from sports_quant.march_madness._config import (
    MM_BACKTEST_DIR,
    MM_MATCHUPS_RESTRUCTURED,
    MM_SCHEDULE_DATA,
    load_models,
)
from sports_quant.march_madness._results import (
    BracketMetrics,
    SimulationMetrics,
    SurvivorMetrics,
    VersionMetrics,
    load_metrics,
    save_metrics,
)
from sports_quant.march_madness.survivor import (
    prepare_game_probs,
    run_survivor_bracket_aware,
    run_survivor_greedy,
    run_survivor_mc_optimal,
    run_survivor_mc_optimal_sequential,
    run_survivor_optimal,
)

logger = logging.getLogger(__name__)

VERSIONS_INDEX = "versions.json"
METRICS_FILE = "metrics.json"


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------


def _collect_bracket_metrics(
    version_dir: Path,
    matchups_df: pd.DataFrame,
    years: list[int],
) -> list[BracketMetrics]:
    """Compute bracket accuracy metrics for each year and source."""
    metrics: list[BracketMetrics] = []

    for year in years:
        year_dir = version_dir / str(year)
        if not year_dir.is_dir():
            logger.warning("Year directory not found: %s", year_dir)
            continue

        actual = build_actual_bracket(matchups_df, year)

        for source, csv_name in [
            ("ensemble", "ensemble_results.csv"),
            ("debiased", "debiased_results.csv"),
        ]:
            csv_path = year_dir / csv_name
            if not csv_path.exists():
                continue

            results_df = pd.read_csv(csv_path)
            predicted = build_predicted_bracket(
                results_df, matchups_df, year, source=source,
            )
            compared = compare_brackets(predicted, actual)
            metrics.append(BracketMetrics.from_bracket(compared))

    return metrics


def _survivor_result_to_metrics(result) -> SurvivorMetrics:
    """Convert a SurvivorResult into a SurvivorMetrics for persistence."""
    picks = tuple(
        {
            "round": p.round_name,
            "day_slot": p.day_slot,
            "team": p.team,
            "seed": p.seed,
            "opponent": p.opponent,
            "win_prob": round(p.win_probability, 4),
            "survived": p.survived,
        }
        for p in result.picks
    )
    return SurvivorMetrics(
        year=result.year,
        strategy=result.strategy,
        rounds_survived=result.rounds_survived,
        survived_all=result.survived_all,
        survival_probability=round(result.survival_probability, 6),
        picks=picks,
        total_rounds=result.total_rounds,
        exhausted=result.exhausted,
    )


def _enrich_matchups_with_day_slots(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Join day_slot data from the schedule CSV onto matchups.

    Returns the matchups unchanged when the schedule file is missing
    or the ``day_slot`` column is already present.
    """
    if "day_slot" in matchups_df.columns:
        return matchups_df

    schedule_path = MM_SCHEDULE_DATA
    if not schedule_path.exists():
        logger.info("No schedule data at %s — skipping day_slot enrichment", schedule_path)
        return matchups_df

    from sports_quant.march_madness.processing.derive_day_slots import (
        derive_day_slots,
        join_day_slots_to_matchups,
    )

    schedule_df = pd.read_csv(schedule_path)
    schedule_with_slots = derive_day_slots(schedule_df)
    return join_day_slots_to_matchups(matchups_df, schedule_with_slots)


def _collect_survivor_metrics(
    version_dir: Path,
    matchups_df: pd.DataFrame,
    years: list[int],
    *,
    feature_lookup=None,
    feature_mode: str = "difference",
) -> list[SurvivorMetrics]:
    """Run survivor strategies for each year and collect metrics.

    When ``feature_lookup`` is provided, also runs bracket-aware and
    MC-optimal strategies that depend on forward simulation.
    """
    from sports_quant.march_madness.simulate import simulate_bracket_deterministic

    metrics: list[SurvivorMetrics] = []

    for year in years:
        year_dir = version_dir / str(year)
        debiased_path = year_dir / "debiased_results.csv"
        if not debiased_path.exists():
            logger.info("No debiased results for %d, skipping survivor", year)
            continue

        debiased_df = pd.read_csv(debiased_path)

        try:
            game_probs = prepare_game_probs(debiased_df, matchups_df, year)
        except ValueError as exc:
            logger.warning("Cannot prepare game probs for %d: %s", year, exc)
            continue

        # Classic strategies (work with actual matchups only)
        for strategy_name, runner in [
            ("greedy", run_survivor_greedy),
            ("optimal", run_survivor_optimal),
        ]:
            result = runner(year, game_probs)
            metrics.append(_survivor_result_to_metrics(result))

        # Bracket-aware strategies (require simulation resources)
        if feature_lookup is None:
            continue

        models_dir = version_dir / str(year) / "models"
        if not models_dir.is_dir():
            logger.info("No models for %d, skipping bracket-aware survivor", year)
            continue

        models = load_models(models_dir)
        if not models:
            logger.warning("No models found for %d", year)
            continue

        actual = build_actual_bracket(matchups_df, year)

        try:
            sim_result = simulate_bracket_deterministic(
                year=year,
                models=models,
                feature_lookup=feature_lookup,
                matchups_df=matchups_df,
                actual_bracket=actual,
                feature_mode=feature_mode,
            )
        except Exception as exc:
            logger.warning(
                "Simulation failed for %d, skipping bracket-aware: %s",
                year, exc,
            )
            continue

        # Bracket-aware (deterministic simulation)
        ba_result = run_survivor_bracket_aware(year, sim_result.bracket, actual)
        metrics.append(_survivor_result_to_metrics(ba_result))

        # MC optimal (averaged across N simulated brackets)
        try:
            mc_result = run_survivor_mc_optimal(
                year=year,
                models=models,
                feature_lookup=feature_lookup,
                matchups_df=matchups_df,
                actual_bracket=actual,
                n_simulations=1000,
                feature_mode=feature_mode,
            )
        except Exception as exc:
            logger.warning(
                "MC optimal failed for %d: %s", year, exc,
            )
            continue
        metrics.append(_survivor_result_to_metrics(mc_result))

        # MC optimal sequential (round-by-round re-optimization)
        try:
            mc_seq_result = run_survivor_mc_optimal_sequential(
                year=year,
                models=models,
                feature_lookup=feature_lookup,
                matchups_df=matchups_df,
                actual_bracket=actual,
                n_simulations=1000,
                feature_mode=feature_mode,
            )
        except Exception as exc:
            logger.warning(
                "MC optimal sequential failed for %d: %s", year, exc,
            )
            continue
        metrics.append(_survivor_result_to_metrics(mc_seq_result))

    return metrics


def _load_simulation_resources(
) -> tuple | None:
    """Load KenPom/Barttorvik data and build a FeatureLookup.

    Returns:
        (feature_lookup, feature_mode) or None if KenPom data missing.
    """
    from sports_quant.march_madness._config import (
        MM_BARTTORVIK_DATA,
        MM_KENPOM_DATA,
        load_mm_config,
    )
    from sports_quant.march_madness._feature_builder import FeatureLookup

    kenpom_path = MM_KENPOM_DATA
    if not kenpom_path.exists():
        logger.warning(
            "KenPom data not found at %s — skipping simulation",
            kenpom_path,
        )
        return None

    cfg = load_mm_config()
    feature_mode = cfg.get("feature_mode", "difference")

    kenpom_df = pd.read_csv(kenpom_path)

    barttorvik_df = None
    bart_path = MM_BARTTORVIK_DATA
    if bart_path.exists() and feature_mode == "combined":
        barttorvik_df = pd.read_csv(bart_path)

    feature_lookup = FeatureLookup(kenpom_df, barttorvik_df=barttorvik_df)
    return feature_lookup, feature_mode


def _collect_simulation_metrics(
    version_dir: Path,
    matchups_df: pd.DataFrame,
    years: list[int],
    *,
    feature_lookup=None,
    feature_mode: str = "difference",
) -> list[SimulationMetrics]:
    """Run forward bracket simulation for each year and collect metrics.

    Loads trained models and KenPom data, then simulates from R64 forward
    (winners cascade) to get realistic bracket accuracy.
    """
    from sports_quant.march_madness.simulate import simulate_bracket_deterministic

    if feature_lookup is None:
        return []

    metrics: list[SimulationMetrics] = []

    for year in years:
        models_dir = version_dir / str(year) / "models"
        if not models_dir.is_dir():
            logger.warning("Models dir not found for %d: %s", year, models_dir)
            continue

        # Load all trained models for ensemble
        models = load_models(models_dir)

        if not models:
            logger.warning("No models found for %d", year)
            continue

        actual = build_actual_bracket(matchups_df, year)

        try:
            result = simulate_bracket_deterministic(
                year=year,
                models=models,
                feature_lookup=feature_lookup,
                matchups_df=matchups_df,
                actual_bracket=actual,
                feature_mode=feature_mode,
            )
        except Exception as exc:
            logger.warning(
                "Simulation failed for %d: %s", year, exc,
            )
            continue

        correct, total = result.overall_accuracy
        accuracy = correct / total if total > 0 else 0.0

        acc_by_round: dict[str, float] = {}
        for round_name, (rc, rt) in result.accuracy_by_round.items():
            acc_by_round[round_name] = rc / rt if rt > 0 else 0.0

        metrics.append(SimulationMetrics(
            year=year,
            overall_correct=correct,
            overall_total=total,
            overall_accuracy=round(accuracy, 4),
            accuracy_by_round=acc_by_round,
        ))

    return metrics


# ---------------------------------------------------------------------------
# Version registry
# ---------------------------------------------------------------------------


def _load_registry(backtest_dir: Path) -> list[dict]:
    """Load the versions index, creating it if absent."""
    index_path = backtest_dir / VERSIONS_INDEX
    if index_path.exists():
        return json.loads(index_path.read_text())
    return []


def _save_registry(backtest_dir: Path, registry: list[dict]) -> None:
    """Write the versions index."""
    index_path = backtest_dir / VERSIONS_INDEX
    index_path.write_text(json.dumps(registry, indent=2) + "\n")


def _update_registry(
    backtest_dir: Path,
    version: str,
    description: str,
    avg_accuracy: float,
    avg_simulation_accuracy: float,
    avg_survivor_rounds: float,
    created_at: str,
) -> list[dict]:
    """Add or update a version entry in the registry."""
    registry = _load_registry(backtest_dir)

    # Remove existing entry for this version if present
    registry = [e for e in registry if e["version"] != version]

    registry.append({
        "version": version,
        "created_at": created_at,
        "description": description,
        "avg_bracket_accuracy": round(avg_accuracy, 4),
        "avg_simulation_accuracy": round(avg_simulation_accuracy, 4),
        "avg_survivor_rounds": round(avg_survivor_rounds, 2),
    })

    # Sort by version name
    registry.sort(key=lambda e: e["version"])
    _save_registry(backtest_dir, registry)
    return registry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_version(
    *,
    version: str,
    description: str,
    backtest_dir: Path | None = None,
    matchups_path: Path | None = None,
    config_path: Path | None = None,
) -> VersionMetrics:
    """Snapshot current backtest results as a named version.

    Computes bracket accuracy and survivor pool metrics for all years,
    writes metrics.json into the version directory, and updates the
    global versions index.

    Args:
        version: Version identifier (e.g. "v1").
        description: Human-readable note about what changed.
        backtest_dir: Override backtest root directory.
        matchups_path: Override matchups CSV path.
        config_path: Override model config path.

    Returns:
        The computed VersionMetrics.
    """
    from sports_quant._config import MODEL_CONFIG_FILE

    backtest_dir = backtest_dir or MM_BACKTEST_DIR
    matchups_path = matchups_path or MM_MATCHUPS_RESTRUCTURED
    config_path = config_path or MODEL_CONFIG_FILE

    version_dir = backtest_dir / version
    if not version_dir.is_dir():
        msg = f"Version directory not found: {version_dir}"
        raise FileNotFoundError(msg)

    # Load config snapshot
    config_snapshot = yaml.safe_load(config_path.read_text())

    # Determine years from subdirectories
    years = sorted(
        int(d.name)
        for d in version_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if not years:
        msg = f"No year directories found in {version_dir}"
        raise FileNotFoundError(msg)

    logger.info(
        "Saving version %s: years=%s, dir=%s", version, years, version_dir,
    )

    # Load matchups and enrich with day_slot from schedule data
    matchups_df = pd.read_csv(matchups_path)
    matchups_df = _enrich_matchups_with_day_slots(matchups_df)

    # Load simulation resources once (shared by simulation + survivor)
    sim_resources = _load_simulation_resources()
    feature_lookup = sim_resources[0] if sim_resources else None
    feature_mode = sim_resources[1] if sim_resources else "difference"

    # Collect metrics
    bracket_metrics = _collect_bracket_metrics(version_dir, matchups_df, years)
    survivor_metrics = _collect_survivor_metrics(
        version_dir, matchups_df, years,
        feature_lookup=feature_lookup,
        feature_mode=feature_mode,
    )
    simulation_metrics = _collect_simulation_metrics(
        version_dir, matchups_df, years,
        feature_lookup=feature_lookup,
        feature_mode=feature_mode,
    )

    created_at = datetime.now(timezone.utc).isoformat()

    vm = VersionMetrics(
        version=version,
        created_at=created_at,
        description=description,
        bracket_metrics=tuple(bracket_metrics),
        survivor_metrics=tuple(survivor_metrics),
        simulation_metrics=tuple(simulation_metrics),
        config_snapshot=config_snapshot,
    )

    # Persist
    save_metrics(vm, version_dir / METRICS_FILE)

    _update_registry(
        backtest_dir,
        version=version,
        description=description,
        avg_accuracy=vm.avg_bracket_accuracy,
        avg_simulation_accuracy=vm.avg_simulation_accuracy,
        avg_survivor_rounds=vm.avg_survivor_rounds,
        created_at=created_at,
    )

    logger.info(
        "Version %s saved. Per-game: %.1f%%, Simulation: %.1f%%, "
        "Survivor avg: %.1f rounds",
        version,
        vm.avg_bracket_accuracy * 100,
        vm.avg_simulation_accuracy * 100,
        vm.avg_survivor_rounds,
    )

    return vm


def list_versions(
    backtest_dir: Path | None = None,
) -> list[dict]:
    """Return all saved versions from the registry.

    Returns:
        List of version summary dicts, sorted by version name.
    """
    backtest_dir = backtest_dir or MM_BACKTEST_DIR
    return _load_registry(backtest_dir)


def compare_versions(
    version_a: str,
    version_b: str,
    *,
    backtest_dir: Path | None = None,
) -> dict:
    """Compare two saved versions side by side.

    Returns a dict with per-year bracket accuracy deltas,
    survivor round deltas, and aggregate summaries.

    Args:
        version_a: First version (baseline).
        version_b: Second version (candidate).
        backtest_dir: Override backtest root.

    Returns:
        Comparison dict with 'bracket', 'survivor', and 'summary' keys.
    """
    backtest_dir = backtest_dir or MM_BACKTEST_DIR

    path_a = backtest_dir / version_a / METRICS_FILE
    path_b = backtest_dir / version_b / METRICS_FILE

    if not path_a.exists():
        msg = f"Metrics not found for {version_a}. Run save-version first."
        raise FileNotFoundError(msg)
    if not path_b.exists():
        msg = f"Metrics not found for {version_b}. Run save-version first."
        raise FileNotFoundError(msg)

    ma = load_metrics(path_a)
    mb = load_metrics(path_b)

    # Bracket comparison: match on (year, source)
    a_bracket = {(m.year, m.source): m for m in ma.bracket_metrics}
    b_bracket = {(m.year, m.source): m for m in mb.bracket_metrics}

    bracket_rows = []
    all_keys = sorted(set(a_bracket.keys()) | set(b_bracket.keys()))
    for key in all_keys:
        ba = a_bracket.get(key)
        bb = b_bracket.get(key)
        acc_a = ba.overall_accuracy if ba else None
        acc_b = bb.overall_accuracy if bb else None
        delta = None
        if acc_a is not None and acc_b is not None:
            delta = acc_b - acc_a
        bracket_rows.append({
            "year": key[0],
            "source": key[1],
            f"{version_a}_accuracy": acc_a,
            f"{version_b}_accuracy": acc_b,
            "delta": delta,
        })

    # Survivor comparison: match on (year, strategy)
    a_surv = {(m.year, m.strategy): m for m in ma.survivor_metrics}
    b_surv = {(m.year, m.strategy): m for m in mb.survivor_metrics}

    survivor_rows = []
    all_surv_keys = sorted(set(a_surv.keys()) | set(b_surv.keys()))
    for key in all_surv_keys:
        sa = a_surv.get(key)
        sb = b_surv.get(key)
        rounds_a = sa.rounds_survived if sa else None
        rounds_b = sb.rounds_survived if sb else None
        delta = None
        if rounds_a is not None and rounds_b is not None:
            delta = rounds_b - rounds_a
        survivor_rows.append({
            "year": key[0],
            "strategy": key[1],
            f"{version_a}_rounds": rounds_a,
            f"{version_b}_rounds": rounds_b,
            "delta": delta,
        })

    # Simulation comparison: match on year
    a_sim = {m.year: m for m in ma.simulation_metrics}
    b_sim = {m.year: m for m in mb.simulation_metrics}

    simulation_rows = []
    all_sim_years = sorted(set(a_sim.keys()) | set(b_sim.keys()))
    for year in all_sim_years:
        sa = a_sim.get(year)
        sb = b_sim.get(year)
        acc_a = sa.overall_accuracy if sa else None
        acc_b = sb.overall_accuracy if sb else None
        delta = None
        if acc_a is not None and acc_b is not None:
            delta = acc_b - acc_a
        simulation_rows.append({
            "year": year,
            f"{version_a}_accuracy": acc_a,
            f"{version_b}_accuracy": acc_b,
            "delta": delta,
        })

    return {
        "version_a": version_a,
        "version_b": version_b,
        "bracket": bracket_rows,
        "simulation": simulation_rows,
        "survivor": survivor_rows,
        "summary": {
            f"{version_a}_avg_accuracy": ma.avg_bracket_accuracy,
            f"{version_b}_avg_accuracy": mb.avg_bracket_accuracy,
            "accuracy_delta": mb.avg_bracket_accuracy - ma.avg_bracket_accuracy,
            f"{version_a}_avg_sim_accuracy": ma.avg_simulation_accuracy,
            f"{version_b}_avg_sim_accuracy": mb.avg_simulation_accuracy,
            "sim_accuracy_delta": (
                mb.avg_simulation_accuracy - ma.avg_simulation_accuracy
            ),
            f"{version_a}_avg_survivor_rounds": ma.avg_survivor_rounds,
            f"{version_b}_avg_survivor_rounds": mb.avg_survivor_rounds,
            "survivor_delta": mb.avg_survivor_rounds - ma.avg_survivor_rounds,
        },
    }
