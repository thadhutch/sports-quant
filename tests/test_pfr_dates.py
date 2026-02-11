from sports_quant.parsers.pfr_dates import extract_date


def test_standard_pfr_title_format():
    title = "Dallas Cowboys at New York Giants - September 26th, 2024"
    assert extract_date(title) == "09/26/2024"


def test_another_date_format():
    title = "Kansas City Chiefs at Buffalo Bills - January 5th, 2025"
    assert extract_date(title) == "01/05/2025"
