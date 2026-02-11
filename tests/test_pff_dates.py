from sports_quant.parsers.pff_dates import extract_date_and_season


def test_regular_season_date():
    result = extract_date_and_season("AC-BB-Sep 10 2024")
    assert result == ("09/10/2024", 2024)


def test_january_year_adjustment():
    """January games belong to the prior season; year should bump by 1."""
    formatted, season = extract_date_and_season("KC-BB-Jan 15 2024")
    assert formatted == "01/15/2025"
    assert season == 2024


def test_february_year_adjustment():
    formatted, season = extract_date_and_season("KC-SF-Feb 11 2024")
    assert formatted == "02/11/2025"
    assert season == 2024


def test_malformed_input_returns_none():
    assert extract_date_and_season("garbage") == (None, None)


def test_two_part_string_returns_none():
    assert extract_date_and_season("AC-BB") == (None, None)
