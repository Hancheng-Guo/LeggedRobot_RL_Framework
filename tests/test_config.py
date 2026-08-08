import pytest
import yaml
from pathlib import Path
from utils.config import load_yaml
from utils.config import get_yaml_value


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


def test_get_yaml_value_dict(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "runner:\n"
        "  params:\n"
        "    rollout_length: 24\n",
        encoding="utf-8",
    )

    value = get_yaml_value(
        path,
        "runner.params.rollout_length",
    )

    assert value == 24


def test_get_yaml_value_string(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "runner:\n"
        "  type: OnPolicyRunner\n",
        encoding="utf-8",
    )

    value = get_yaml_value(
        path,
        "runner.type",
    )

    assert value == "OnPolicyRunner"


def test_get_yaml_value_list(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "robots:\n"
        "  - name: a1\n"
        "    mass: 12\n"
        "  - name: go1\n"
        "    mass: 15\n",
        encoding="utf-8",
    )

    value = get_yaml_value(
        path,
        "robots.1.mass",
    )

    assert value == 15


def test_get_yaml_value_missing_with_default(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "runner:\n"
        "  params:\n"
        "    rollout_length: 24\n",
        encoding="utf-8",
    )

    value = get_yaml_value(
        path,
        "runner.params.batch_size",
        default=64,
    )

    assert value == 64


def test_get_yaml_value_missing_required(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "runner:\n"
        "  params:\n"
        "    rollout_length: 24\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        get_yaml_value(
            path,
            "runner.params.batch_size",
            required=True,
        )


def test_get_yaml_value_invalid_list_index(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "robots:\n"
        "  - name: a1\n"
        "    mass: 12\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        get_yaml_value(
            path,
            "robots.first.mass",
            required=True,
        )


def test_get_yaml_value_list_out_of_range(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "robots:\n"
        "  - name: a1\n"
        "    mass: 12\n",
        encoding="utf-8",
    )

    with pytest.raises(IndexError):
        get_yaml_value(
            path,
            "robots.10.mass",
            required=True,
        )


def test_get_yaml_value_access_non_container(tmp_path):

    path = tmp_path / "tmp.yaml"

    path.write_text(
        "runner:\n"
        "  type: OnPolicyRunner\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        get_yaml_value(
            path,
            "runner.type.name",
            required=True,
        )