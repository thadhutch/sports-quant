"""Tests for March Madness feature definitions."""

from sports_quant.march_madness._features import (
    DROP_COLUMNS,
    TEAM_INFO_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    KENPOM_COLUMNS,
    TEAM_NAME_MAPPING,
    standardize_team_name,
)


def test_drop_columns_are_strings():
    assert all(isinstance(c, str) for c in DROP_COLUMNS)
    assert len(DROP_COLUMNS) > 0


def test_team_info_columns_are_strings():
    assert all(isinstance(c, str) for c in TEAM_INFO_COLUMNS)


def test_target_column():
    assert TARGET_COLUMN == "Team1_Win"


def test_year_column():
    assert YEAR_COLUMN == "YEAR"


def test_kenpom_columns_count():
    assert len(KENPOM_COLUMNS) == 22


def test_team_name_mapping_values():
    assert TEAM_NAME_MAPPING["N.C. State"] == "North Carolina St."
    assert TEAM_NAME_MAPPING["Louisiana Lafayette"] == "Louisiana"
    assert TEAM_NAME_MAPPING["College of Charleston"] == "Charleston"


def test_standardize_team_name_mapped():
    assert standardize_team_name("N.C. State") == "North Carolina St."


def test_standardize_team_name_unmapped():
    assert standardize_team_name("Duke") == "Duke"


def test_drop_columns_no_target():
    """TARGET_COLUMN should not be in DROP_COLUMNS for training data prep."""
    assert TARGET_COLUMN not in DROP_COLUMNS
