"""Tests for March Madness KenPom data cleaning pipeline."""

import csv

from sports_quant.march_madness.processing.clean_kenpom import (
    clean_data,
    fix_duplicate_ranks,
    merge_school_names,
    remove_seed_numbers,
    fix_split_names,
    _is_text,
)


def test_is_text_true():
    assert _is_text("Duke") is True
    assert _is_text("North Carolina") is True


def test_is_text_false():
    assert _is_text("123") is False
    assert _is_text("45.6") is False


def test_clean_data(tmp_path):
    """Removes empty values and strips whitespace."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Team", "", "Conference"])
        writer.writerow(["1", " Duke ", "", "ACC"])
        writer.writerow(["2", "Kansas", "", "B12"])

    clean_data(input_path, output_path)

    with open(output_path, "r") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["Rank", "Team", "Conference"]
    assert rows[1] == ["1", "Duke", "ACC"]


def test_fix_duplicate_ranks(tmp_path):
    """Removes duplicate rank from second column."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Team", "Conference"])
        writer.writerow(["1", "1", "Duke"])  # Duplicate rank as 2nd col
        writer.writerow(["2", "Kansas", "B12"])  # Normal

    fix_duplicate_ranks(input_path, output_path)

    with open(output_path, "r") as f:
        rows = list(csv.reader(f))

    assert rows[1][1] == "Duke"  # Duplicate rank removed, shifted
    assert rows[2][1] == "Kansas"  # Unchanged


def test_merge_school_names(tmp_path):
    """Merges split school name columns."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Team", "Conference", "Record"])
        writer.writerow(["1", "North", "Carolina", "30.5"])  # Split name (text+text)
        writer.writerow(["2", "Kansas", "25.0", "28.7"])  # Normal (text+number)

    merge_school_names(input_path, output_path)

    with open(output_path, "r") as f:
        rows = list(csv.reader(f))

    # "North" and "Carolina" should be merged (both text)
    assert rows[1][1] == "North Carolina"
    # "Kansas" and "25.0" should NOT be merged (text+number)
    assert rows[2][1] == "Kansas"


def test_merge_school_names_skips_known_conference(tmp_path):
    """Regression: conference abbreviations must NOT be merged into team name.

    Without this guard, 'Mount St. Mary's' + 'MAAC' would merge into
    'Mount St. Mary's MAAC', shifting all subsequent columns and corrupting
    the team's stats (AdjEM ~101 instead of ~-6).
    """
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Team", "Conference", "Record"])
        writer.writerow(["238", "Mount St. Mary's", "MAAC", "23-12"])
        writer.writerow(["185", "Merrimack", "MAAC", "18-15"])

    merge_school_names(input_path, output_path)

    with open(output_path, "r") as f:
        rows = list(csv.reader(f))

    # Conference abbreviation must stay separate, not merged into team name
    assert rows[1][1] == "Mount St. Mary's"
    assert rows[1][2] == "MAAC"
    assert rows[2][1] == "Merrimack"
    assert rows[2][2] == "MAAC"


def test_remove_seed_numbers(tmp_path):
    """Removes seed number column after team name."""
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    with open(input_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Team", "Conference"])
        writer.writerow(["1", "Duke", "1", "ACC"])  # Has seed number
        writer.writerow(["2", "Kansas", "B12", "28-7"])  # No seed

    remove_seed_numbers(input_path, output_path)

    with open(output_path, "r") as f:
        rows = list(csv.reader(f))

    # Seed number "1" should be removed
    assert rows[1] == ["1", "Duke", "ACC"]
