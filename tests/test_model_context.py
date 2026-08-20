from pathlib import Path

import mujoco

from envs.simulators.mujoco import MujocoSimulator


def test_mujoco_builds_observation_indices_from_model(runtime_context):
    model_path = (
        Path(__file__).parents[1]
        / "assets"
        / "unitree_go1"
        / "MJCF"
        / "go1.xml"
    )
    simulator = MujocoSimulator(runtime_context)
    simulator.models = [
        mujoco.MjModel.from_xml_path(str(model_path))   # pyright: ignore[reportAttributeAccessIssue]
    ]

    simulator._build_model_context()
    context = simulator.model_context

    assert context.base_pos_qpos_ids.tolist() == [0, 1, 2]
    assert context.base_quat_qpos_ids.tolist() == [3, 4, 5, 6]
    assert context.base_ang_vel_qvel_ids.tolist() == [3, 4, 5]
    assert context.joint_qpos_ids.tolist() == list(range(7, 19))
    assert context.joint_qvel_ids.tolist() == list(range(6, 18))
