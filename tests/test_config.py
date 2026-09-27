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


def test_generated_files_are_git_ignored():
    """Build outputs must never be committed (regression: the artifact metadata JSON
    was once tracked, so every `python -m churn.train` left the repository dirty)."""
    from fnmatch import fnmatch

    from churn.train import artifact_paths

    patterns = [
        line.strip() for line in (config.PROJECT_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith(("#", "!"))
    ]
    paths = artifact_paths()
    generated = [
        paths.model, paths.metadata, config.RAW_DATA_FILE,
        config.PROCESSED_DATA_DIR / "oof_baseline_predictions.csv",
    ]
    for path in generated:
        relative = path.relative_to(config.PROJECT_ROOT).as_posix()
        assert any(fnmatch(relative, pattern) for pattern in patterns), f"{relative} is not git-ignored"
