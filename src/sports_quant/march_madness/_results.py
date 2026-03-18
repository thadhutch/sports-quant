"""Metrics dataclasses for bracket and survivor pool results.

Provides immutable, serializable structures for capturing model performance
across backtest years. Used by the versioning system to persist and compare
results across model iterations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sports_quant.march_madness._bracket import Bracket, ROUND_ORDER


# ---------------------------------------------------------------------------
# Bracket metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BracketMetrics:
    """Accuracy metrics for a single bracket (one year, one source)."""

    year: int
    source: str
    overall_accuracy: float
    accuracy_by_round: dict[str, float]
    total_games: int
    correct_games: int
    upset_count: int

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "source": self.source,
            "overall_accuracy": self.overall_accuracy,
            "accuracy_by_round": self.accuracy_by_round,
            "total_games": self.total_games,
            "correct_games": self.correct_games,
            "upset_count": self.upset_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BracketMetrics:
        return cls(
            year=d["year"],
            source=d["source"],
            overall_accuracy=d["overall_accuracy"],
            accuracy_by_round=d["accuracy_by_round"],
            total_games=d["total_games"],
            correct_games=d["correct_games"],
            upset_count=d["upset_count"],
        )

    @classmethod
    def from_bracket(cls, bracket: Bracket) -> BracketMetrics:
        """Extract metrics from a compared Bracket (with is_correct flags)."""
        evaluated = [g for g in bracket.games if g.is_correct is not None]
        correct = sum(1 for g in evaluated if g.is_correct)
        total = len(evaluated)
        overall = correct / total if total > 0 else 0.0

        return cls(
            year=bracket.year,
            source=bracket.source,
            overall_accuracy=overall,
            accuracy_by_round=bracket.accuracy_by_round(),
            total_games=total,
            correct_games=correct,
            upset_count=bracket.upset_count(),
        )


# ---------------------------------------------------------------------------
# Survivor metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivorMetrics:
    """Metrics for a single survivor pool run (one year, one strategy)."""

    year: int
    strategy: str
    rounds_survived: int
    survived_all: bool
    survival_probability: float
    picks: tuple[dict, ...]
    total_rounds: int = 6
    exhausted: bool = False  # True when no candidate was available

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "strategy": self.strategy,
            "rounds_survived": self.rounds_survived,
            "survived_all": self.survived_all,
            "survival_probability": self.survival_probability,
            "picks": list(self.picks),
            "total_rounds": self.total_rounds,
            "exhausted": self.exhausted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SurvivorMetrics:
        return cls(
            year=d["year"],
            strategy=d["strategy"],
            rounds_survived=d["rounds_survived"],
            survived_all=d["survived_all"],
            survival_probability=d["survival_probability"],
            picks=tuple(d["picks"]),
            total_rounds=d.get("total_rounds", 6),
            exhausted=d.get("exhausted", False),
        )


# ---------------------------------------------------------------------------
# Simulation metrics (forward R64→NCG)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationMetrics:
    """Accuracy metrics from forward bracket simulation (R64→NCG).

    Unlike BracketMetrics which evaluates per-game predictions on known
    matchups, this captures the realistic bracket accuracy where wrong
    picks cascade into wrong future matchups.
    """

    year: int
    overall_correct: int
    overall_total: int
    overall_accuracy: float
    accuracy_by_round: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "overall_correct": self.overall_correct,
            "overall_total": self.overall_total,
            "overall_accuracy": self.overall_accuracy,
            "accuracy_by_round": self.accuracy_by_round,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SimulationMetrics:
        return cls(
            year=d["year"],
            overall_correct=d["overall_correct"],
            overall_total=d["overall_total"],
            overall_accuracy=d["overall_accuracy"],
            accuracy_by_round=d["accuracy_by_round"],
        )


# ---------------------------------------------------------------------------
# Version metrics (aggregated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionMetrics:
    """Complete metrics for a model version across all backtest years."""

    version: str
    created_at: str
    description: str
    bracket_metrics: tuple[BracketMetrics, ...]
    survivor_metrics: tuple[SurvivorMetrics, ...]
    simulation_metrics: tuple[SimulationMetrics, ...]
    config_snapshot: dict

    @property
    def avg_bracket_accuracy(self) -> float:
        """Average overall bracket accuracy across all years/sources."""
        if not self.bracket_metrics:
            return 0.0
        return sum(
            m.overall_accuracy for m in self.bracket_metrics
        ) / len(self.bracket_metrics)

    @property
    def avg_simulation_accuracy(self) -> float:
        """Average forward-simulation accuracy across all years."""
        if not self.simulation_metrics:
            return 0.0
        return sum(
            m.overall_accuracy for m in self.simulation_metrics
        ) / len(self.simulation_metrics)

    @property
    def avg_survivor_rounds(self) -> float:
        """Average rounds survived across all years/strategies."""
        if not self.survivor_metrics:
            return 0.0
        return sum(
            m.rounds_survived for m in self.survivor_metrics
        ) / len(self.survivor_metrics)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "bracket": [m.to_dict() for m in self.bracket_metrics],
            "survivor": [m.to_dict() for m in self.survivor_metrics],
            "simulation": [m.to_dict() for m in self.simulation_metrics],
            "config": self.config_snapshot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VersionMetrics:
        return cls(
            version=d["version"],
            created_at=d["created_at"],
            description=d["description"],
            bracket_metrics=tuple(
                BracketMetrics.from_dict(b) for b in d.get("bracket", [])
            ),
            survivor_metrics=tuple(
                SurvivorMetrics.from_dict(s) for s in d.get("survivor", [])
            ),
            simulation_metrics=tuple(
                SimulationMetrics.from_dict(s)
                for s in d.get("simulation", [])
            ),
            config_snapshot=d.get("config", {}),
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_metrics(metrics: VersionMetrics, path: Path) -> Path:
    """Write VersionMetrics to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
    return path


def load_metrics(path: Path) -> VersionMetrics:
    """Read VersionMetrics from a JSON file."""
    data = json.loads(path.read_text())
    return VersionMetrics.from_dict(data)
