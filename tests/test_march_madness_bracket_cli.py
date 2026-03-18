"""Tests for March Madness bracket CLI integration."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from sports_quant.cli import cli
from sports_quant.march_madness._bracket import Bracket, BracketGame, BracketSlot
from sports_quant.march_madness._bracket_cli import (
    resolve_backtest_paths,
    render_year_brackets,
    render_multi_year_accuracy,
    run_bracket_visualisation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mini_bracket(year: int, source: str = "actual") -> Bracket:
    """Create a minimal 1-game bracket for testing."""
    game = BracketGame(
        round_name="NCG",
        region=-1,
        game_index=0,
        team1=BracketSlot(team="Duke", seed=1),
        team2=BracketSlot(team="UNC", seed=2),
        winner=BracketSlot(team="Duke", seed=1),
        win_probability=0.75,
        is_upset=False,
        is_correct=None,
    )
    return Bracket(year=year, source=source, games=(game,))


# ---------------------------------------------------------------------------
# resolve_backtest_paths
# ---------------------------------------------------------------------------


class TestResolveBacktestPaths:
    def test_returns_expected_keys(self, tmp_path: Path):
        version_dir = tmp_path / "v1" / "2024"
        version_dir.mkdir(parents=True)
        (version_dir / "ensemble_results.csv").touch()
        (version_dir / "debiased_results.csv").touch()

        paths = resolve_backtest_paths(
            backtest_dir=tmp_path, version="v1", year=2024,
        )

        assert paths["year"] == 2024
        assert paths["ensemble"].name == "ensemble_results.csv"
        assert paths["debiased"].name == "debiased_results.csv"
        assert paths["output_dir"] == version_dir / "plots" / "brackets"

    def test_raises_for_missing_year_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="2024"):
            resolve_backtest_paths(
                backtest_dir=tmp_path, version="v1", year=2024,
            )

    def test_raises_for_missing_ensemble_csv(self, tmp_path: Path):
        version_dir = tmp_path / "v1" / "2024"
        version_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="ensemble_results.csv"):
            resolve_backtest_paths(
                backtest_dir=tmp_path, version="v1", year=2024,
            )


# ---------------------------------------------------------------------------
# render_year_brackets
# ---------------------------------------------------------------------------


class TestRenderYearBrackets:
    @patch("sports_quant.march_madness._bracket_cli.render_comparison")
    @patch("sports_quant.march_madness._bracket_cli.render_bracket")
    @patch("sports_quant.march_madness._bracket_cli.build_predicted_bracket")
    @patch("sports_quant.march_madness._bracket_cli.build_actual_bracket")
    def test_renders_all_sources(
        self, mock_actual, mock_predicted, mock_render, mock_compare, tmp_path,
    ):
        actual = _make_mini_bracket(2024, "actual")
        ensemble = _make_mini_bracket(2024, "ensemble")
        debiased = _make_mini_bracket(2024, "debiased")

        mock_actual.return_value = actual
        mock_predicted.side_effect = [ensemble, debiased]
        mock_render.side_effect = lambda _b, path, **kw: Path(path)
        mock_compare.side_effect = lambda _p, _a, path, **kw: Path(path)

        matchups_df = pd.DataFrame()
        ensemble_df = pd.DataFrame()
        debiased_df = pd.DataFrame()

        results = render_year_brackets(
            year=2024,
            matchups_df=matchups_df,
            ensemble_df=ensemble_df,
            debiased_df=debiased_df,
            output_dir=tmp_path,
        )

        # Should render: actual, ensemble, debiased, ensemble_comparison, debiased_comparison
        assert len(results) == 5
        assert mock_actual.call_count == 1
        assert mock_predicted.call_count == 2
        assert mock_render.call_count == 3  # actual + ensemble + debiased
        assert mock_compare.call_count == 2  # ensemble vs actual + debiased vs actual

    @patch("sports_quant.march_madness._bracket_cli.render_comparison")
    @patch("sports_quant.march_madness._bracket_cli.render_bracket")
    @patch("sports_quant.march_madness._bracket_cli.build_predicted_bracket")
    @patch("sports_quant.march_madness._bracket_cli.build_actual_bracket")
    def test_single_source(
        self, mock_actual, mock_predicted, mock_render, mock_compare, tmp_path,
    ):
        actual = _make_mini_bracket(2024, "actual")
        ensemble = _make_mini_bracket(2024, "ensemble")

        mock_actual.return_value = actual
        mock_predicted.return_value = ensemble
        mock_render.side_effect = lambda _b, path, **kw: Path(path)
        mock_compare.side_effect = lambda _p, _a, path, **kw: Path(path)

        results = render_year_brackets(
            year=2024,
            matchups_df=pd.DataFrame(),
            ensemble_df=pd.DataFrame(),
            debiased_df=None,
            output_dir=tmp_path,
        )

        # actual + ensemble + ensemble_comparison = 3
        assert len(results) == 3
        assert mock_predicted.call_count == 1


# ---------------------------------------------------------------------------
# render_multi_year_accuracy
# ---------------------------------------------------------------------------


class TestRenderMultiYearAccuracy:
    @patch("sports_quant.march_madness._bracket_cli.render_accuracy_comparison")
    @patch("sports_quant.march_madness._bracket_cli.render_accuracy")
    def test_renders_per_year_and_multi(
        self, mock_accuracy, mock_multi, tmp_path,
    ):
        brackets = {
            2023: _make_mini_bracket(2023, "ensemble_vs_actual"),
            2024: _make_mini_bracket(2024, "ensemble_vs_actual"),
        }
        output_dirs = {
            2023: tmp_path / "2023" / "plots",
            2024: tmp_path / "2024" / "plots",
        }
        mock_accuracy.side_effect = lambda _b, path, **kw: Path(path)
        mock_multi.side_effect = lambda _bs, path, **kw: Path(path)

        results = render_multi_year_accuracy(
            compared_brackets=brackets,
            output_dirs=output_dirs,
            multi_year_output_dir=tmp_path,
        )

        assert mock_accuracy.call_count == 2  # one per year
        assert mock_multi.call_count == 1     # one multi-year chart
        assert len(results) == 3


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestBracketCLI:
    def test_bracket_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["march-madness", "bracket", "--help"])
        assert result.exit_code == 0
        assert "bracket" in result.output.lower()

    def test_year_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["march-madness", "bracket", "--help"])
        assert "--year" in result.output

    def test_source_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["march-madness", "bracket", "--help"])
        assert "--source" in result.output

    @patch("sports_quant.march_madness._bracket_cli.run_bracket_visualisation")
    def test_invokes_orchestrator(self, mock_run):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["march-madness", "bracket", "--year", "2024"],
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["years"] == [2024]

    @patch("sports_quant.march_madness._bracket_cli.run_bracket_visualisation")
    def test_multiple_years(self, mock_run):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["march-madness", "bracket", "--year", "2023", "--year", "2024"],
        )
        assert result.exit_code == 0
        assert mock_run.call_args[1]["years"] == [2023, 2024]

    @patch("sports_quant.march_madness._bracket_cli.run_bracket_visualisation")
    def test_default_runs_all_years(self, mock_run):
        runner = CliRunner()
        result = runner.invoke(cli, ["march-madness", "bracket"])
        assert result.exit_code == 0
        # Should use config backtest_years when no --year specified
        mock_run.assert_called_once()

    @patch("sports_quant.march_madness._bracket_cli.run_bracket_visualisation")
    def test_source_filter(self, mock_run):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["march-madness", "bracket", "--source", "debiased"],
        )
        assert result.exit_code == 0
        assert mock_run.call_args[1]["sources"] == ["debiased"]
