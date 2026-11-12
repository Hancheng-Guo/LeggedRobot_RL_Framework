"""Optional managers disable cleanly without hiding invalid configurations."""
import pytest
import torch

from envs.tasks.locomotion import LocomotionTaskLogic
from utils import Component


OPTIONAL_CONFIGS = (
    "curriculum_manager_config", "event_manager_config", "randomization_manager_config",
)


def required_config():
    return {
        "constants": {}, "task_state_manager_config": {},
        "action_manager_config": {"terms": {}},
        "command_manager_config": {"terms": {"speed": {
            "type": "Uniform", "params": {"min_value": 1., "max_value": 1.},
        }}},
        "observation_manager_config": {"terms": {"base_height": {}}},
        "reward_manager_config": {"terms": {}},
        "termination_manager_config": {"terms": {}},
    }


def configure(context, model_context, config):
    task = LocomotionTaskLogic(context)
    task.prepare(config=config)
    task.config_update(Component(None, None, None, None, None, None),
                       num_envs=3, model_context=model_context, **config)
    return task


@pytest.mark.parametrize("form", ["omitted", "null", "empty"])
def test_optional_managers_can_be_disabled(runtime_context, model_context, form):
    config = required_config()
    if form != "omitted":
        config.update({name: None if form == "null" else {} for name in OPTIONAL_CONFIGS})
    task = configure(runtime_context, model_context, config)
    assert task.curriculum_manager.terms == {}
    assert task.randomization_manager.terms == {}
    assert task.get_simulator_requirements() == {}
    task.reset()
    torch.testing.assert_close(task.command["speed"], torch.ones(3, 1))


@pytest.mark.parametrize("form", ["omitted", "null", "empty"])
def test_missing_referenced_curriculum_is_an_error(runtime_context, model_context, form):
    config = required_config()
    config["command_manager_config"]["terms"]["speed"] = {
        "type": "Curriculum", "params": {
            "curriculum": "lpac_reward",
            "min_value": 0., "max_value": 1., "num_cell": 2,
        },
    }
    if form != "omitted":
        config["curriculum_manager_config"] = None if form == "null" else {}
    with pytest.raises(ValueError, match="unknown curriculum"):
        configure(runtime_context, model_context, config)


@pytest.mark.parametrize("name", OPTIONAL_CONFIGS)
@pytest.mark.parametrize("value", [[], False, 0, "invalid"])
def test_invalid_optional_config_is_not_silently_disabled(runtime_context, model_context, name, value):
    config = required_config()
    config[name] = value
    error = NotImplementedError if name == "event_manager_config" else (TypeError, ValueError)
    with pytest.raises(error):
        configure(runtime_context, model_context, config)


def test_unimplemented_event_config_is_rejected(runtime_context, model_context):
    config = required_config()
    config["event_manager_config"] = {"terms": {"unsupported": {}}}
    with pytest.raises(NotImplementedError, match="Event manager"):
        configure(runtime_context, model_context, config)
