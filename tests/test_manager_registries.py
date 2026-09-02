from envs.tasks.managers.action.terms.registry import ACTION_CLASS_MAP
from envs.tasks.managers.observation.terms.registry import (
    OBSERVATION_CLASS_MAP,
)
from envs.tasks.managers.reward.terms.registry import REWARD_CLASS_MAP
from envs.tasks.managers.termination.terms.registry import (
    TERMINATION_CLASS_MAP,
)
from envs.tasks.managers.curriculum.terms.registry import (
    CURRICULUM_CLASS_MAP,
)
from utils.string import camel_to_snake


def test_camel_to_snake_handles_terms_and_acronyms():
    assert camel_to_snake("HardClamp") == "hard_clamp"
    assert camel_to_snake("BaseAngularVelocity") == "base_angular_velocity"
    assert camel_to_snake("ActionDiffL2") == "action_diff_l2"
    assert camel_to_snake("HTTPServer") == "h_t_t_p_server"
    assert camel_to_snake("XYZ") == "x_y_z"
    assert camel_to_snake("TrackZVel") == "track_z_vel"
    assert camel_to_snake("TrackXyVel") == "track_xy_vel"
    assert camel_to_snake("TrackXYVel") == "track_x_y_vel"


def test_manager_registries_use_snake_case_keys():
    assert ACTION_CLASS_MAP["hard_clamp"].__name__ == "HardClamp"
    assert ACTION_CLASS_MAP["linear_map"].__name__ == "LinearMap"
    assert (
        OBSERVATION_CLASS_MAP["base_angular_velocity"].__name__
        == "BaseAngularVelocity"
    )
    assert OBSERVATION_CLASS_MAP["last_action"].__name__ == "LastAction"
    assert REWARD_CLASS_MAP["action_diff_l2"].__name__ == "ActionDiffL2"
    assert TERMINATION_CLASS_MAP["base_height"].__name__ == "BaseHeight"
    assert (
        CURRICULUM_CLASS_MAP["lrpc_command_reward"].__name__
        == "LrpcCommandReward"
    )
