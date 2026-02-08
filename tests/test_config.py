import os
from pathlib import Path


def test_config_paths_are_path_objects():
    import config

    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.PFF_DATA_DIR, Path)
    assert isinstance(config.PFR_DATA_DIR, Path)
    assert isinstance(config.OVERUNDER_DATA_DIR, Path)


def test_config_default_values():
    import config

    assert config.MAX_WEEK == int(os.environ.get("NFL_MAX_WEEK", "18"))
    assert isinstance(config.SEASONS, list)


def test_config_env_var_override(monkeypatch):
    monkeypatch.setenv("NFL_MAX_WEEK", "10")
    monkeypatch.setenv("NFL_SEASONS", "2022,2023")

    # Force reimport to pick up new env vars
    import importlib
    import config

    importlib.reload(config)

    assert config.MAX_WEEK == 10
    assert config.SEASONS == ["2022", "2023"]
