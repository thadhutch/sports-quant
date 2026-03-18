"""Tests for March Madness versioning system."""

import json
from pathlib import Path

import pytest

from sports_quant.march_madness._results import (
    BracketMetrics,
    SurvivorMetrics,
    VersionMetrics,
    save_metrics,
)
from sports_quant.march_madness._versioning import (
    _load_registry,
    _save_registry,
    _update_registry,
    compare_versions,
    list_versions,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_load_empty(self, tmp_path: Path):
        result = _load_registry(tmp_path)
        assert result == []

    def test_save_and_load(self, tmp_path: Path):
        entries = [{"version": "v1", "description": "test"}]
        _save_registry(tmp_path, entries)
        loaded = _load_registry(tmp_path)
        assert loaded == entries

    def test_update_adds_entry(self, tmp_path: Path):
        registry = _update_registry(
            tmp_path,
            version="v1",
            description="baseline",
            avg_accuracy=0.73,
            avg_simulation_accuracy=0.65,
            avg_survivor_rounds=3.5,
            created_at="2026-03-15T00:00:00",
        )
        assert len(registry) == 1
        assert registry[0]["version"] == "v1"
        assert registry[0]["avg_bracket_accuracy"] == 0.73

    def test_update_replaces_existing(self, tmp_path: Path):
        _update_registry(
            tmp_path, version="v1", description="first",
            avg_accuracy=0.70, avg_simulation_accuracy=0.60, avg_survivor_rounds=3.0,
            created_at="2026-03-15T00:00:00",
        )
        registry = _update_registry(
            tmp_path, version="v1", description="updated",
            avg_accuracy=0.75, avg_simulation_accuracy=0.68, avg_survivor_rounds=4.0,
            created_at="2026-03-15T01:00:00",
        )
        assert len(registry) == 1
        assert registry[0]["description"] == "updated"
        assert registry[0]["avg_bracket_accuracy"] == 0.75

    def test_update_sorts_by_version(self, tmp_path: Path):
        _update_registry(
            tmp_path, version="v2", description="second",
            avg_accuracy=0.75, avg_simulation_accuracy=0.68, avg_survivor_rounds=4.0,
            created_at="2026-03-15T00:00:00",
        )
        registry = _update_registry(
            tmp_path, version="v1", description="first",
            avg_accuracy=0.70, avg_simulation_accuracy=0.60, avg_survivor_rounds=3.0,
            created_at="2026-03-15T01:00:00",
        )
        assert registry[0]["version"] == "v1"
        assert registry[1]["version"] == "v2"


# ---------------------------------------------------------------------------
# Compare versions
# ---------------------------------------------------------------------------


def _save_test_metrics(
    tmp_path: Path,
    version: str,
    accuracy: float,
    rounds_survived: int,
) -> None:
    vm = VersionMetrics(
        version=version,
        created_at="2026-03-15T00:00:00",
        description=f"{version} test",
        bracket_metrics=(
            BracketMetrics(
                year=2024,
                source="ensemble_vs_actual",
                overall_accuracy=accuracy,
                accuracy_by_round={"R64": accuracy},
                total_games=63,
                correct_games=int(63 * accuracy),
                upset_count=5,
            ),
        ),
        survivor_metrics=(
            SurvivorMetrics(
                year=2024,
                strategy="greedy",
                rounds_survived=rounds_survived,
                survived_all=rounds_survived == 6,
                survival_probability=0.5,
                picks=(),
            ),
        ),
        simulation_metrics=(),
        config_snapshot={},
    )
    version_dir = tmp_path / version
    version_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(vm, version_dir / "metrics.json")


class TestCompareVersions:
    def test_compare(self, tmp_path: Path):
        _save_test_metrics(tmp_path, "v1", accuracy=0.70, rounds_survived=3)
        _save_test_metrics(tmp_path, "v2", accuracy=0.75, rounds_survived=5)

        result = compare_versions("v1", "v2", backtest_dir=tmp_path)

        assert result["version_a"] == "v1"
        assert result["version_b"] == "v2"

        # Bracket delta
        assert len(result["bracket"]) == 1
        row = result["bracket"][0]
        assert row["delta"] == pytest.approx(0.05)

        # Survivor delta
        assert len(result["survivor"]) == 1
        assert result["survivor"][0]["delta"] == 2

        # Summary
        assert result["summary"]["accuracy_delta"] == pytest.approx(0.05)
        assert result["summary"]["survivor_delta"] == 2

    def test_compare_missing_version(self, tmp_path: Path):
        _save_test_metrics(tmp_path, "v1", accuracy=0.70, rounds_survived=3)
        with pytest.raises(FileNotFoundError, match="v99"):
            compare_versions("v1", "v99", backtest_dir=tmp_path)

    def test_compare_asymmetric_years(self, tmp_path: Path):
        """Versions with different year coverage still compare correctly."""
        # v1 has 2024 data
        _save_test_metrics(tmp_path, "v1", accuracy=0.70, rounds_survived=3)

        # v2 has different accuracy
        _save_test_metrics(tmp_path, "v2", accuracy=0.80, rounds_survived=6)

        result = compare_versions("v1", "v2", backtest_dir=tmp_path)
        assert result["summary"]["accuracy_delta"] == pytest.approx(0.10)


class TestListVersions:
    def test_empty(self, tmp_path: Path):
        result = list_versions(backtest_dir=tmp_path)
        assert result == []

    def test_with_entries(self, tmp_path: Path):
        _update_registry(
            tmp_path, version="v1", description="test",
            avg_accuracy=0.7, avg_simulation_accuracy=0.6, avg_survivor_rounds=3.0,
            created_at="2026-03-15T00:00:00",
        )
        result = list_versions(backtest_dir=tmp_path)
        assert len(result) == 1
        assert result[0]["version"] == "v1"
