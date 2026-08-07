from pathlib import Path

from app.utils.component import create_component


def test_create_component():

    component_config = {
        "runner": {
            "type": "on_policy",
            "config_name": "on_policy",
        },
        "algorithm": {
            "type": "ppo",
            "config_name": "ppo",
        },
        "model": {
            "type": "actor_critic",
            "config_name": "actor_critic",
        },
        "simulator": {
            "type": "mujoco",
            "config_name": "mujoco",
        },
        "task": {
            "type": "locomotion",
            "config_name": "locomotion",
        },
    }

    component = create_component(component_config)

    assert component.runner.type == "on_policy"
    assert component.algorithm.type == "ppo"
    assert component.model.type == "actor_critic"
    assert component.simulator.type == "mujoco"
    assert component.task.type == "locomotion"

    assert component.runner.config == Path("./configs/runner/on_policy.yaml")
    assert component.algorithm.config == Path("./configs/algorithm/ppo.yaml")
    assert component.model.config == Path("./configs/model/actor_critic.yaml")
    assert component.simulator.config == Path("./configs/simulator/mujoco.yaml")
    assert component.task.config == Path("./configs/task/locomotion.yaml")


def test_component_use_config_path():

    component_config = {
        "runner": {
            "type": "on_policy",
            "config_path": "./abc/custom.yaml",
        },
        "algorithm": {
            "type": "ppo",
            "config_name": "ppo",
        },
        "model": {
            "type": "actor_critic",
            "config_name": "actor_critic",
        },
        "simulator": {
            "type": "mujoco",
            "config_name": "mujoco",
        },
        "task": {
            "type": "locomotion",
            "config_name": "locomotion",
        },
    }

    component = create_component(component_config)

    assert component.runner.config == Path("./abc/custom.yaml")


def test_append_yaml_suffix():

    component_config = {
        "runner": {
            "type": "on_policy",
            "config_path": "./configs/runner/on_policy",
        },
        "algorithm": {
            "type": "ppo",
            "config_name": "ppo",
        },
        "model": {
            "type": "actor_critic",
            "config_name": "actor_critic",
        },
        "simulator": {
            "type": "mujoco",
            "config_name": "mujoco",
        },
        "task": {
            "type": "locomotion",
            "config_name": "locomotion",
        },
    }

    component = create_component(component_config)

    assert component.runner.config.suffix == ".yaml"