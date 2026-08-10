from pathlib import Path

from utils.component import create_component


def test_create_component():

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

    component = create_component(component_config)

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
        "./configs/envs/vector_env.yaml"
    )

    assert component.simulator.config == Path(
        "./configs/simulators/mujoco.yaml"
    )

    assert component.task.config == Path(
        "./configs/tasks/locomotion.yaml"
    )


def test_component_use_config_path():

    component_config = {
        "runner": {
            "type": "on_policy",
            "config_path": "./abc/custom.yaml",
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

    component = create_component(component_config)

    assert component.runner.config == Path(
        "./abc/custom.yaml"
    )


def test_append_yaml_suffix():

    component_config = {
        "runner": {
            "type": "on_policy",
            "config_path": "./configs/runner/on_policy",
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

    component = create_component(component_config)

    assert component.runner.config.suffix == ".yaml"