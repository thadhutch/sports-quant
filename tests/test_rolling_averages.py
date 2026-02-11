from sports_quant.processing.rolling_averages import (
    initialize_team_stats,
    calculate_avg_stats,
    stat_columns,
)


def test_initialize_team_stats_all_zero():
    stats = initialize_team_stats()
    assert all(v == 0.0 for v in stats.values())
    assert set(stats.keys()) == set(stat_columns)


def test_calculate_avg_stats_zero_games():
    stats = initialize_team_stats()
    avg = calculate_avg_stats(stats, 0)
    assert all(v == 0.0 for v in avg.values())


def test_calculate_avg_stats_after_n_games():
    stats = {col: 10.0 for col in stat_columns}
    avg = calculate_avg_stats(stats, 5)
    assert all(v == 2.0 for v in avg.values())


def test_calculate_avg_stats_single_game():
    stats = {col: 75.3 for col in stat_columns}
    avg = calculate_avg_stats(stats, 1)
    assert all(v == 75.3 for v in avg.values())
