"""Tests for March Madness results metrics and versioning."""

import json
from pathlib import Path

import pytest

from sports_quant.march_madness._bracket import (
    Bracket,
    BracketGame,
    BracketSlot,
)
from sports_quant.march_madness._results import (
    BracketMetrics,
    SimulationMetrics,
    SurvivorMetrics,
    VersionMetrics,
    load_metrics,
    save_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game(*, correct: bool | None = True, upset: bool = False) -> BracketGame:
    return BracketGame(
        round_name="R64",
        region=0,
        game_index=0,
        team1=BracketSlot(team="Duke", seed=1),
        team2=BracketSlot(team="UNC", seed=16),
        winner=BracketSlot(team="Duke", seed=1),
        win_probability=0.95,
        is_upset=upset,
        is_correct=correct,
    )


def _make_bracket(correct_count: int, total: int) -> Bracket:
    games = []
    for i in range(total):
        games.append(BracketGame(
            round_name="R64" if i < 32 else "R32",
            region=i % 4,
            game_index=i,
            team1=BracketSlot(team=f"Team{i}A", seed=1),
            team2=BracketSlot(team=f"Team{i}B", seed=16),
            winner=BracketSlot(team=f"Team{i}A", seed=1),
            win_probability=0.9,
            is_upset=False,
            is_correct=i < correct_count,
        ))
    return Bracket(year=2024, source="ensemble_vs_actual", games=tuple(games))


# ---------------------------------------------------------------------------
# BracketMetrics
# ---------------------------------------------------------------------------


class TestBracketMetrics:
    def test_from_bracket(self):
        bracket = _make_bracket(correct_count=8, total=10)
        metrics = BracketMetrics.from_bracket(bracket)

        assert metrics.year == 2024
        assert metrics.source == "ensemble_vs_actual"
        assert metrics.overall_accuracy == 0.8
        assert metrics.total_games == 10
        assert metrics.correct_games == 8

    def test_round_trip(self):
        metrics = BracketMetrics(
            year=2023,
            source="debiased_vs_actual",
            overall_accuracy=0.75,
            accuracy_by_round={"R64": 0.8, "R32": 0.6},
            total_games=63,
            correct_games=47,
            upset_count=12,
        )
        restored = BracketMetrics.from_dict(metrics.to_dict())
        assert restored == metrics

    def test_from_bracket_empty(self):
        bracket = Bracket(year=2024, source="test", games=())
        metrics = BracketMetrics.from_bracket(bracket)
        assert metrics.overall_accuracy == 0.0
        assert metrics.total_games == 0


# ---------------------------------------------------------------------------
# SurvivorMetrics
# ---------------------------------------------------------------------------


class TestSurvivorMetrics:
    def test_round_trip(self):
        metrics = SurvivorMetrics(
            year=2024,
            strategy="greedy",
            rounds_survived=4,
            survived_all=False,
            survival_probability=0.65,
            picks=(
                {"round": "R64", "team": "Duke", "seed": 1, "survived": True},
            ),
        )
        restored = SurvivorMetrics.from_dict(metrics.to_dict())
        assert restored == metrics


# ---------------------------------------------------------------------------
# SimulationMetrics
# ---------------------------------------------------------------------------


class TestSimulationMetrics:
    def test_round_trip(self):
        metrics = SimulationMetrics(
            year=2024,
            overall_correct=42,
            overall_total=63,
            overall_accuracy=0.6667,
            accuracy_by_round={"R64": 0.75, "R32": 0.625},
        )
        restored = SimulationMetrics.from_dict(metrics.to_dict())
        assert restored == metrics


# ---------------------------------------------------------------------------
# VersionMetrics
# ---------------------------------------------------------------------------


class TestVersionMetrics:
    def test_avg_bracket_accuracy(self):
        vm = VersionMetrics(
            version="v1",
            created_at="2026-03-15T00:00:00",
            description="test",
            bracket_metrics=(
                BracketMetrics(
                    year=2024, source="ensemble", overall_accuracy=0.70,
                    accuracy_by_round={}, total_games=63, correct_games=44,
                    upset_count=5,
                ),
                BracketMetrics(
                    year=2024, source="debiased", overall_accuracy=0.80,
                    accuracy_by_round={}, total_games=63, correct_games=50,
                    upset_count=6,
                ),
            ),
            survivor_metrics=(),
            simulation_metrics=(),
            config_snapshot={},
        )
        assert vm.avg_bracket_accuracy == pytest.approx(0.75)

    def test_avg_survivor_rounds(self):
        vm = VersionMetrics(
            version="v1",
            created_at="2026-03-15T00:00:00",
            description="test",
            bracket_metrics=(),
            survivor_metrics=(
                SurvivorMetrics(
                    year=2024, strategy="greedy", rounds_survived=3,
                    survived_all=False, survival_probability=0.5, picks=(),
                ),
                SurvivorMetrics(
                    year=2024, strategy="optimal", rounds_survived=5,
                    survived_all=False, survival_probability=0.3, picks=(),
                ),
            ),
            simulation_metrics=(),
            config_snapshot={},
        )
        assert vm.avg_survivor_rounds == pytest.approx(4.0)

    def test_round_trip(self):
        vm = VersionMetrics(
            version="v2",
            created_at="2026-03-15T12:00:00",
            description="Added tempo features",
            bracket_metrics=(
                BracketMetrics(
                    year=2024, source="ensemble", overall_accuracy=0.73,
                    accuracy_by_round={"R64": 0.8}, total_games=63,
                    correct_games=46, upset_count=8,
                ),
            ),
            survivor_metrics=(
                SurvivorMetrics(
                    year=2024, strategy="greedy", rounds_survived=4,
                    survived_all=False, survival_probability=0.6,
                    picks=({"round": "R64", "team": "Duke"},),
                ),
            ),
            simulation_metrics=(),
            config_snapshot={"march_madness": {"model_version": "v2"}},
        )
        restored = VersionMetrics.from_dict(vm.to_dict())
        assert restored.version == vm.version
        assert restored.avg_bracket_accuracy == vm.avg_bracket_accuracy

    def test_empty_metrics(self):
        vm = VersionMetrics(
            version="v0",
            created_at="2026-01-01T00:00:00",
            description="empty",
            bracket_metrics=(),
            survivor_metrics=(),
            simulation_metrics=(),
            config_snapshot={},
        )
        assert vm.avg_bracket_accuracy == 0.0
        assert vm.avg_survivor_rounds == 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path):
        vm = VersionMetrics(
            version="v1",
            created_at="2026-03-15T00:00:00",
            description="baseline",
            bracket_metrics=(
                BracketMetrics(
                    year=2024, source="ensemble", overall_accuracy=0.72,
                    accuracy_by_round={"R64": 0.8, "R32": 0.6},
                    total_games=63, correct_games=45, upset_count=7,
                ),
            ),
            survivor_metrics=(),
            simulation_metrics=(),
            config_snapshot={"march_madness": {"model_version": "v1"}},
        )

        path = tmp_path / "metrics.json"
        save_metrics(vm, path)

        assert path.exists()
        loaded = load_metrics(path)
        assert loaded.version == "v1"
        assert loaded.bracket_metrics[0].overall_accuracy == pytest.approx(0.72)
        assert loaded.config_snapshot == vm.config_snapshot

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "metrics.json"
        vm = VersionMetrics(
            version="v1", created_at="", description="",
            bracket_metrics=(), survivor_metrics=(), simulation_metrics=(),
            config_snapshot={},
        )
        save_metrics(vm, path)
        assert path.exists()
