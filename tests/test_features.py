"""Tests for shared feature definitions."""

from sports_quant.modeling._features import (
    ALL_FEATURES,
    DISPLAY_NAMES,
    DROP_COLUMNS,
    RANK_FEATURES,
)


def test_rank_features_count():
    # 11 PFF categories x 2 (home/away) = 22
    assert len(RANK_FEATURES) == 22


def test_all_features_includes_ou_line():
    assert "ou_line" in ALL_FEATURES
    assert len(ALL_FEATURES) == len(RANK_FEATURES) + 1


def test_display_names_cover_all_features():
    for feat in ALL_FEATURES:
        assert feat in DISPLAY_NAMES, f"Missing display name for {feat}"


def test_drop_columns_excludes_features():
    """DROP_COLUMNS should not overlap with ALL_FEATURES."""
    overlap = set(DROP_COLUMNS) & set(ALL_FEATURES)
    assert not overlap, f"DROP_COLUMNS overlaps with ALL_FEATURES: {overlap}"


def test_rank_features_are_paired():
    """Every home rank feature should have a matching away rank feature."""
    home = [f for f in RANK_FEATURES if f.startswith("home-")]
    away = [f for f in RANK_FEATURES if f.startswith("away-")]
    assert len(home) == len(away)
    for h in home:
        expected_away = h.replace("home-", "away-", 1)
        assert expected_away in away, f"Missing away counterpart for {h}"
