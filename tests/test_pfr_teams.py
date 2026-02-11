from sports_quant.parsers.pfr_teams import extract_teams


def test_standard_away_at_home_format():
    title = "Dallas Cowboys at New York Giants - September 26th, 2024"
    away, home = extract_teams(title)
    assert away == "Dallas Cowboys"
    assert home == "New York Giants"


def test_quoted_title():
    title = '"Green Bay Packers at Chicago Bears - October 6th, 2024'
    away, home = extract_teams(title)
    assert away == "Green Bay Packers"
    assert home == "Chicago Bears"
