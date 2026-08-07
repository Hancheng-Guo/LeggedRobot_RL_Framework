import pytest
from pathlib import Path

from app.utils.path import fill_path


def test_fill_path_with_name_only():
    result = fill_path(
        file_name="config.yaml"
    )

    assert isinstance(result, Path)
    assert result == Path("config.yaml")


def test_fill_path_with_directory():
    result = fill_path(
        file_name="config.yaml",
        file_dir="configs"
    )

    assert result == Path("configs/config.yaml")


def test_fill_path_with_path():
    result = fill_path(
        file_path="configs/config.yaml"
    )

    assert result == Path("configs/config.yaml")


def test_file_path_override_warning():
    with pytest.warns(UserWarning):
        result = fill_path(
            file_name="old.yaml",
            file_dir="old_dir",
            file_path="new.yaml",
        )

    assert result == Path("new.yaml")


def test_missing_file_name():
    with pytest.raises(ValueError):
        fill_path()