import pytest
import yaml
from utils.config import load_yaml


def test_load_yaml_success(tmp_path):

    expected_config = {
        "seed": 1,
        "device": "cpu",
        "deterministic": True,
    }

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.dump(expected_config),
        encoding="utf-8",
    )

    config = load_yaml(path)

    assert config == expected_config

def test_load_yaml_success_with_str(tmp_path):

    expected_config = {
        "seed": 1,
        "device": "cpu",
        "deterministic": True,
    }

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.dump(expected_config),
        encoding="utf-8",
    )

    config = load_yaml(str(path))

    assert config == expected_config


def test_load_yaml_file_not_found():

    with pytest.raises(FileNotFoundError):
        load_yaml("not_exist.yaml")


def test_load_yaml_empty_file(tmp_path):

    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_yaml(path)


def test_load_yaml_root_not_dict(tmp_path):

    path = tmp_path / "list.yaml"
    path.write_text(
        "- a\n"
        "- b\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        load_yaml(path)