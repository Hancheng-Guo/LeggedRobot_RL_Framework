from envs.tasks.managers.action.terms.registry import ACTION_CLASS_MAP
from envs.tasks.managers.observation.terms.registry import (
    OBSERVATION_CLASS_MAP,
)
from envs.tasks.managers.reward.terms.registry import REWARD_CLASS_MAP
from utils.string import camel_to_snake


def test_camel_to_snake_handles_terms_and_acronyms():
    assert camel_to_snake("HardClamp") == "hard_clamp"
    assert camel_to_snake("BaseAngularVelocity") == "base_angular_velocity"
    assert camel_to_snake("ActionDiffL2") == "action_diff_l2"
    assert camel_to_snake("HTTPServer") == "http_server"


def test_manager_registries_use_snake_case_keys():
    assert ACTION_CLASS_MAP["hard_clamp"].__name__ == "HardClamp"
    assert ACTION_CLASS_MAP["linear_map"].__name__ == "LinearMap"
    assert (
        OBSERVATION_CLASS_MAP["base_angular_velocity"].__name__
        == "BaseAngularVelocity"
    )
    assert OBSERVATION_CLASS_MAP["last_action"].__name__ == "LastAction"
    assert REWARD_CLASS_MAP["action_diff_l2"].__name__ == "ActionDiffL2"

