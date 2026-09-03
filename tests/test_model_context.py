from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from envs.simulators.mujoco import MujocoSimulator
from utils.component import Component


def _configure_go1_simulator(
    runtime_context,
    *,
    num_envs: int = 1,
    reset_keyframe: str | None = None,
) -> MujocoSimulator:
    model_path = (
        Path(__file__).parents[1]
        / "assets"
        / "unitree_go1"
        / "MJCF"
        / "go1.xml"
    )
    simulator = MujocoSimulator(runtime_context)
    simulator.config_update(
        component=Component(None, None, None, None, None, None),
        num_envs=num_envs,
        model_path=model_path,
        sim_dt=0.002,
        frame_skip=1,
        geom_foot_names=(),
        geom_floor_names=(),
        reset_keyframe=reset_keyframe,
    )
    return simulator


def test_mujoco_builds_observation_indices_from_model(runtime_context):
    model_path = (
        Path(__file__).parents[1]
        / "assets"
        / "unitree_go1"
        / "MJCF"
        / "go1.xml"
    )
    simulator = MujocoSimulator(runtime_context)
    simulator.geom_foot_names = ("FR", "FL", "RR", "RL")
    simulator.geom_floor_names = ()
    simulator.models = [
        mujoco.MjModel.from_xml_path(str(model_path))   # pyright: ignore[reportAttributeAccessIssue]
    ]

    simulator._build_model_context()
    context = simulator.model_context

    assert context.base_id == context.body_names.index("trunk")
    assert context.base_pos_qpos_ids.tolist() == [0, 1, 2]
    assert context.base_quat_qpos_ids.tolist() == [3, 4, 5, 6]
    assert context.base_lin_vel_qvel_ids.tolist() == [0, 1, 2]
    assert context.base_ang_vel_qvel_ids.tolist() == [3, 4, 5]
    assert context.joint_qpos_ids.tolist() == list(range(7, 19))
    assert context.joint_qvel_ids.tolist() == list(range(6, 18))
    assert context.joint_default_pos.shape == (12,)
    assert context.geom_foot_ids.tolist() == [
        context.geom_names.index(name)
        for name in ("FR", "FL", "RR", "RL")
    ]
    assert "trunk" in context.body_names
    assert context.geom_names[context.geom_names.index("FR")] == "FR"


def test_mujoco_scene_exposes_ground_and_body_geom_mapping(runtime_context):
    model_path = (
        Path(__file__).parents[1]
        / "assets"
        / "unitree_go1"
        / "MJCF"
        / "scene.xml"
    )
    simulator = MujocoSimulator(runtime_context)
    simulator.geom_foot_names = ()
    simulator.geom_floor_names = ("floor",)
    simulator.models = [
        mujoco.MjModel.from_xml_path(str(model_path))  # pyright: ignore[reportAttributeAccessIssue]
    ]

    simulator._build_model_context()
    context = simulator.model_context

    floor_geom_id = context.geom_names.index("floor")
    floor_body_id = int(context.geom_body_ids[floor_geom_id])

    assert context.body_names[floor_body_id] == "world"
    assert "FR_thigh" in context.body_names


def test_mujoco_state_exposes_base_velocity_in_body_frame(runtime_context):
    model_path = (
        Path(__file__).parents[1]
        / "assets"
        / "unitree_go1"
        / "MJCF"
        / "go1.xml"
    )
    simulator = MujocoSimulator(runtime_context)
    simulator.geom_foot_names = ()
    simulator.geom_floor_names = ()
    simulator.models = [
        mujoco.MjModel.from_xml_path(str(model_path))  # pyright: ignore[reportAttributeAccessIssue]
    ]
    simulator.datas = [
        mujoco.MjData(simulator.models[0])  # pyright: ignore[reportAttributeAccessIssue]
    ]

    simulator._build_model_context()
    mujoco.mj_forward(  # pyright: ignore[reportAttributeAccessIssue]
        simulator.models[0],
        simulator.datas[0],
    )

    state = simulator.get_state()

    assert state["base_lin_vel_body"].shape == (1, 3)
    assert state["base_ang_vel_body"].shape == (1, 3)


def test_mujoco_reset_uses_configured_keyframe(runtime_context):
    simulator = _configure_go1_simulator(
        runtime_context,
        reset_keyframe="home",
    )
    model = simulator.models[0]
    data = simulator.datas[0]
    keyframe_id = simulator.reset_keyframe_id

    data.qpos[:] = 0.0
    data.qvel[:] = 1.0
    data.ctrl[:] = 0.0
    data.time = 1.0
    simulator.reset()

    np.testing.assert_allclose(data.qpos, model.key_qpos[keyframe_id])
    np.testing.assert_allclose(data.qvel, model.key_qvel[keyframe_id])
    np.testing.assert_allclose(data.ctrl, model.key_ctrl[keyframe_id])
    assert data.time == pytest.approx(model.key_time[keyframe_id])
    np.testing.assert_allclose(
        simulator.model_context.joint_default_pos.cpu().numpy(),
        model.key_qpos[
            keyframe_id,
            simulator.model_context.joint_qpos_ids.cpu().numpy(),
        ],
    )


def test_mujoco_reset_only_changes_selected_environments(runtime_context):
    simulator = _configure_go1_simulator(
        runtime_context,
        num_envs=2,
        reset_keyframe="home",
    )
    untouched_qpos = simulator.datas[1].qpos.copy()
    simulator.datas[0].qpos[:] = 0.0
    simulator.datas[1].qpos[:] += 0.25
    modified_untouched_qpos = simulator.datas[1].qpos.copy()

    simulator.reset(torch.tensor([0]))

    np.testing.assert_allclose(
        simulator.datas[0].qpos,
        simulator.models[0].key_qpos[simulator.reset_keyframe_id],
    )
    np.testing.assert_allclose(
        simulator.datas[1].qpos,
        modified_untouched_qpos,
    )
    assert not np.allclose(untouched_qpos, modified_untouched_qpos)


def test_mujoco_reset_without_keyframe_uses_model_defaults(runtime_context):
    simulator = _configure_go1_simulator(runtime_context)
    model = simulator.models[0]
    data = simulator.datas[0]
    data.qpos[:] = 0.0
    data.qvel[:] = 1.0

    simulator.reset()

    assert simulator.reset_keyframe_id == -1
    np.testing.assert_allclose(data.qpos, model.qpos0)
    np.testing.assert_allclose(data.qvel, 0.0)


def test_mujoco_rejects_unknown_reset_keyframe(runtime_context):
    with pytest.raises(ValueError, match="Keyframe 'missing' was not found"):
        _configure_go1_simulator(
            runtime_context,
            reset_keyframe="missing",
        )


def test_mujoco_reconfiguration_does_not_reuse_stale_keyframe_id(
    runtime_context,
):
    simulator = _configure_go1_simulator(
        runtime_context,
        reset_keyframe="home",
    )
    model_path = simulator.model_path

    simulator.config_update(
        component=Component(None, None, None, None, None, None),
        model_path=model_path,
    )
    simulator.reset()

    assert simulator.reset_keyframe is None
    assert simulator.reset_keyframe_id == -1
    np.testing.assert_allclose(
        simulator.datas[0].qpos,
        simulator.models[0].qpos0,
    )
