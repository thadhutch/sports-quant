from sports_quant.parsers.pff_teams import map_teams


def test_known_abbreviation_pair():
    team_0, team_1 = map_teams("AC-BB-Sep 10 2024")
    assert team_0 == "Arizona Cardinals"
    assert team_1 == "Buffalo Bills"


def test_unknown_abbreviation_falls_back_to_raw():
    team_0, team_1 = map_teams("ZZ-XX-Oct 1 2024")
    assert team_0 == "ZZ"
    assert team_1 == "XX"
