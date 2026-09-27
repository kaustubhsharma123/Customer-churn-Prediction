"""Smoke tests: the package is installed and its configured paths are valid."""

from churn import config


def test_project_root_contains_pyproject():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_configured_directories_exist():
    for directory in (
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.FIGURES_DIR,
    ):
        assert directory.is_dir(), f"Missing directory: {directory}"


def test_random_state_is_int():
    assert isinstance(config.RANDOM_STATE, int)
