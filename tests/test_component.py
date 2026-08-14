from pathlib import Path

from utils.component import create_component


def test_create_component():

    load_dir = Path(".")

    component_config = {
        "runner": {
            "type": "on_policy",
            "config": "on_policy",
        },
        "algorithm": {
            "type": "ppo",
            "config": "ppo",
        },
        "model": {
            "type": "actor_critic",
            "config": "actor_critic",
        },
        "environment": {
            "type": "vector_env",
            "config": "vector_env",
        },
        "simulator": {
            "type": "mujoco",
            "config": "mujoco",
        },
        "task": {
            "type": "locomotion",
            "config": "locomotion",
        },
    }

    component = create_component(component_config, load_dir)

    assert component.runner.type == "on_policy"
    assert component.algorithm.type == "ppo"
    assert component.model.type == "actor_critic"
    assert component.environment.type == "vector_env"
    assert component.simulator.type == "mujoco"
    assert component.task.type == "locomotion"

    assert component.runner.config == Path(
        "./configs/runners/on_policy.yaml"
    )

    assert component.algorithm.config == Path(
        "./configs/algorithms/ppo.yaml"
    )

    assert component.model.config == Path(
        "./configs/models/actor_critic.yaml"
    )

    assert component.environment.config == Path(
        "./configs/environments/vector_env.yaml"
    )

    assert component.simulator.config == Path(
        "./configs/simulators/mujoco.yaml"
    )

    assert component.task.config == Path(
        "./configs/tasks/locomotion.yaml"
    )


def test_component_use_config_path():

    load_dir = Path(".")

    component_config = {
        "runner": {
            "type": "on_policy",
            "config_path": "./abc/custom.yaml",
        },
    }

    component = create_component(component_config, load_dir)

    assert component.runner is not None
    assert component.runner.config == Path("abc/custom.yaml")


def test_append_yaml_suffix():

    load_dir = Path(".")

    component_config = {
        "runner": {
            "type": "on_policy",
            "config": "on_policy",
        },
    }

    component = create_component(component_config, load_dir)

    assert component.runner is not None
    assert component.runner.config == Path(
        "configs/runners/on_policy.yaml"
    )
    assert component.runner.config.suffix == ".yaml"


def test_missing_component_is_none():

    load_dir = Path(".")

    component_config = {
        "runner": {
            "type": "on_policy",
            "config": "on_policy",
        },
    }

    component = create_component(component_config, load_dir)

    assert component.runner is not None
    assert component.algorithm is None
    assert component.model is None
    assert component.environment is None
    assert component.simulator is None
    assert component.task is None


def test_component_use_config_dir():

    load_dir = Path(".")

    component_config = {
        "runner": {
            "type": "on_policy",
            "config": "custom",
            "config_dir": "custom_configs",
        },
    }

    component = create_component(component_config, load_dir)

    assert component.runner is not None
    assert component.runner.config == Path(
        "custom_configs/custom.yaml"
    )


def test_load_dir_is_used_for_default_config_dir():

    load_dir = Path("experiment")

    component_config = {
        "algorithm": {
            "type": "ppo",
            "config": "ppo",
        },
    }

    component = create_component(component_config, load_dir)

    assert component.algorithm is not None
    assert component.algorithm.config == Path(
        "experiment/configs/algorithms/ppo.yaml"
    )