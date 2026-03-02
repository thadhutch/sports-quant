"""KenPom data cleaning pipeline.

Consolidates the raw_kenpom/ cleaning steps into a single module:
  1. convert_to_csv — parse raw text into CSV
  2. clean_data — remove empty values
  3. fix_duplicate_ranks — remove duplicate rank columns
  4. merge_school_names — merge split school name columns
  5. remove_seed_numbers — strip seed number columns
  6. fix_split_names — fix names split across columns
  7. append_kenpom_data — combine historical + new data
"""

import csv
import logging
from pathlib import Path

import pandas as pd

from sports_quant.march_madness import _config as mm_config

logger = logging.getLogger(__name__)

# Known conference abbreviations for identifying split team names
KNOWN_CONFERENCES = [
    "ACC", "SEC", "B10", "B12", "BE", "Amer", "A10", "WCC", "MWC", "MVC",
    "Ivy", "SB", "CUSA", "ASun", "BW", "CAA", "Horz", "BSth", "SC", "WAC",
    "Slnd", "AE", "MAC", "OVC", "BSky", "Sum", "PL", "MEAC", "SWAC", "NEC",
]


def convert_to_csv(input_path: Path, output_path: Path) -> None:
    """Convert a raw text file of KenPom data into CSV format."""
    headers = [
        "Rank", "Team", "Conference", "Record", "AdjEM", "AdjO", "AdjO_Rank",
        "AdjD", "AdjD_Rank", "AdjT", "AdjT_Rank", "Luck", "Luck_Rank",
        "SOS_AdjEM", "SOS_AdjEM_Rank", "OppO", "OppO_Rank", "OppD",
        "OppD_Rank", "NCSOS_AdjEM", "NCSOS_AdjEM_Rank", "Year",
    ]

    with open(input_path, "r") as infile, open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)

        for line in infile:
            if not line.strip():
                continue
            parts = line.strip().split()
            rank = parts[0]
            data_columns = parts[-21:]
            team_name = " ".join(parts[1:-21])
            row = [rank, team_name] + data_columns
            writer.writerow(row)

    logger.info("Converted text to CSV: %s -> %s", input_path, output_path)


def clean_data(input_path: Path, output_path: Path) -> None:
    """Remove empty values and strip whitespace from CSV rows."""
    with open(input_path, "r") as infile:
        rows = list(csv.reader(infile))

    cleaned_rows = []
    for row in rows:
        cleaned = [value.strip() for value in row if value.strip()]
        cleaned_rows.append(cleaned)

    with open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)

    logger.info("Cleaned data: %s -> %s", input_path, output_path)


def fix_duplicate_ranks(input_path: Path, output_path: Path) -> None:
    """Fix rows with duplicate rank values."""

    def _fix_row(row: list[str]) -> list[str]:
        if len(row) >= 2 and row[0].isdigit():
            rank = row[0]
            if row[1].startswith(f"{rank},"):
                row[1] = row[1].split(",", 1)[1]
            elif len(row) > 2 and row[1] == rank:
                row.pop(1)
        return row

    with open(input_path, "r") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = [_fix_row(row) for row in reader]

    with open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("Fixed duplicate ranks: %s -> %s", input_path, output_path)


def _is_text(value: str) -> bool:
    """Check if a value is text (not numeric)."""
    try:
        float(value)
        return False
    except ValueError:
        return True


def merge_school_names(input_path: Path, output_path: Path) -> None:
    """Merge consecutive text columns that represent split school names."""
    with open(input_path, "r") as infile:
        rows = list(csv.reader(infile))

    processed_rows = []
    for row in rows:
        if len(row) >= 3 and _is_text(row[1]) and _is_text(row[2]):
            merged_name = f"{row[1]} {row[2]}"
            new_row = [row[0], merged_name] + row[3:]
        else:
            new_row = row
        processed_rows.append(new_row)

    with open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

    logger.info("Merged school names: %s -> %s", input_path, output_path)


def remove_seed_numbers(input_path: Path, output_path: Path) -> None:
    """Remove seed number columns that appear after team names."""
    with open(input_path, "r") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        processed_rows = [header]
        for row in reader:
            team_index = 1
            if len(row) > team_index + 1 and row[team_index + 1].isdigit():
                row.pop(team_index + 1)
            processed_rows.append(row)

    with open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

    logger.info("Removed seed numbers: %s -> %s", input_path, output_path)


def fix_split_names(input_path: Path, output_path: Path) -> None:
    """Fix team names incorrectly split across columns."""
    with open(input_path, "r") as infile, open(output_path, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            if len(row) >= 4 and row[2] not in KNOWN_CONFERENCES:
                row[1] = row[1] + " " + row[2]
                row.pop(2)
            writer.writerow(row)

    logger.info("Fixed split names: %s -> %s", input_path, output_path)


def append_kenpom_data(
    existing_path: Path,
    new_path: Path,
    output_path: Path,
    year: int = 2025,
) -> None:
    """Append new KenPom data to existing historical data."""
    existing_data = pd.read_csv(existing_path)
    new_data = pd.read_csv(new_path)
    new_data["Year"] = year

    combined = pd.concat([existing_data, new_data], ignore_index=True)
    combined.to_csv(output_path, index=False)

    logger.info(
        "Appended %d rows (year %d) to %d existing rows -> %s (%d total)",
        len(new_data), year, len(existing_data), output_path, len(combined),
    )


def run_kenpom_cleaning_pipeline(
    raw_input: Path | None = None,
    final_output: Path | None = None,
) -> None:
    """Run the full KenPom data cleaning pipeline.

    Chains all cleaning steps using intermediate files in the cleaning_steps
    directory. The pipeline expects a raw text file as input and produces a
    clean CSV as output.

    Args:
        raw_input: Path to the raw KenPom text file.
        final_output: Path for the final cleaned CSV output.
    """
    cleaning_dir = mm_config.MM_CLEANING_DIR
    cleaning_dir.mkdir(parents=True, exist_ok=True)

    raw_input = raw_input or (cleaning_dir / "kenpom_data.txt")
    final_output = final_output or (cleaning_dir / "kenpom_data_final.csv")

    # Step 1: Text -> CSV
    step1 = cleaning_dir / "kenpom_data_new.csv"
    convert_to_csv(raw_input, step1)

    # Step 2: Clean empty values
    step2 = cleaning_dir / "kenpom_data_cleaned.csv"
    clean_data(step1, step2)

    # Step 3: Fix duplicate ranks
    step3 = cleaning_dir / "kenpom_data_deduped.csv"
    fix_duplicate_ranks(step2, step3)

    # Step 4: Merge school names
    step4 = cleaning_dir / "kenpom_data_merged.csv"
    merge_school_names(step3, step4)

    # Step 5: Remove seed numbers
    step5 = cleaning_dir / "kenpom_data_seedless.csv"
    remove_seed_numbers(step4, step5)

    # Step 6: Fix split names
    fix_split_names(step5, final_output)

    logger.info("KenPom cleaning pipeline complete: %s", final_output)


if __name__ == "__main__":
    run_kenpom_cleaning_pipeline()
