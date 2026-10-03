import mujoco
from mujoco import viewer
import pytest
import torch
import numpy as np
from contextlib import nullcontext
from pathlib import Path

from envs.simulators.mujoco import MujocoSimulator
from envs.simulators.utils import SimulatorState
from utils import Component


pytestmark = pytest.mark.mujoco


def _configure_go1_simulator(
    runtime_context,
    *,
    num_envs: int = 1,
    reset_keyframe: str | None = None,
    frame_skip: int = 1,
    step_workers: int = 1,
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
        frame_skip=frame_skip,
        step_workers=step_workers,
        foot_geom_names=(),
        floor_geom_names=(),
        reset_keyframe=reset_keyframe,
    )
    return simulator


def test_mujoco_rejects_unsupported_render_mode(runtime_context):
    simulator = _configure_go1_simulator(runtime_context)

    try:
        with pytest.raises(ValueError, match="Unsupported render mode"):
            simulator.config_update(
                component=Component(None, None, None, None, None, None),
                render_mode="depth_array",
            )
    finally:
        simulator.close()


def test_mujoco_viewer_tracks_base_without_changing_view(runtime_context, monkeypatch):
    simulator = _configure_go1_simulator(runtime_context)
    camera = mujoco.MjvCamera()  # pyright: ignore[reportAttributeAccessIssue]
    camera.azimuth = 35.0
    camera.elevation = -20.0
    camera.distance = 4.0

    class FakeViewer:
        cam = camera

        def lock(self):
            return nullcontext()

        def sync(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(viewer, "launch_passive", lambda *_: FakeViewer())

    try:
        simulator._human_render()
        assert camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING  # pyright: ignore[reportAttributeAccessIssue]
        assert camera.trackbodyid == simulator.model_context.base_id
        assert (camera.azimuth, camera.elevation, camera.distance) == (
            35.0, -20.0, 4.0,
        )
    finally:
        simulator.close()


def test_mujoco_recording_camera_follows_base(runtime_context, monkeypatch):
    simulator = _configure_go1_simulator(runtime_context)
    cameras = []

    class FakeRenderer:
        def __init__(self, model):
            self.model = model

        def update_scene(self, data, camera):
            cameras.append(camera)

        def render(self):
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)

    try:
        simulator._rgb_array_render()
        camera = cameras[0]
        assert camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING  # pyright: ignore[reportAttributeAccessIssue]
        assert camera.trackbodyid == simulator.model_context.base_id

        model, data = simulator.models[0], simulator.datas[0]
        scene = mujoco.MjvScene(model, maxgeom=1000)  # pyright: ignore[reportAttributeAccessIssue]
        option = mujoco.MjvOption()  # pyright: ignore[reportAttributeAccessIssue]

        def camera_position():
            mujoco.mjv_updateScene(  # pyright: ignore[reportAttributeAccessIssue]
                model, data, option, None, camera,
                mujoco.mjtCatBit.mjCAT_ALL, scene,  # pyright: ignore[reportAttributeAccessIssue]
            )
            return scene.camera[0].pos.copy()

        before = camera_position()
        data.qpos[0] += 2.0
        mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        after = camera_position()
        assert after[0] > before[0] + 1.5
    finally:
        simulator.close()


def test_mujoco_rejects_actuator_activation_states(runtime_context):
    model = mujoco.MjModel.from_xml_string( # pyright: ignore[reportAttributeAccessIssue]
        """
        <mujoco>
          <worldbody>
            <body name="base">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
              <body name="link">
                <joint name="joint" type="hinge"/>
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <general joint="joint" dyntype="filter" dynprm="0.1"/>
          </actuator>
        </mujoco>
        """
    )
    simulator = MujocoSimulator(runtime_context)
    simulator.models = [model]

    with pytest.raises(
        NotImplementedError,
        match="actuator activation states are not supported",
    ):
        simulator._build_model_context()


def test_mujoco_builds_observation_indices_from_model(runtime_context):
    model_path = (
        Path(__file__).parents[1]
        / "assets"
        / "unitree_go1"
        / "MJCF"
        / "go1.xml"
    )
    simulator = MujocoSimulator(runtime_context)
    simulator.foot_geom_names = ("FR", "FL", "RR", "RL")
    simulator.floor_geom_names = ()
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
    assert context.foot_geom_ids.tolist() == [
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
    simulator.foot_geom_names = ()
    simulator.floor_geom_names = ("floor",)
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
    simulator.foot_geom_names = ()
    simulator.floor_geom_names = ()
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

    assert isinstance(state, SimulatorState)

    assert state.base_lin_vel_body.shape == (1, 3)
    assert state.base_ang_vel_body.shape == (1, 3)
    assert state.foot_ground_contact.shape == (
        1,
        simulator.model_context.foot_geom_ids.numel(),
    )


def test_mujoco_foot_ground_contact_applies_force_threshold(
    runtime_context,
    model_context,
):
    simulator = MujocoSimulator(runtime_context)
    simulator.model_context = model_context
    simulator.foot_contact_force_threshold = 15.0
    contact_geom_ids = torch.tensor([[
        [3, 0],
        [3, 0],
        [0, 3],
        [1, 0],
    ]])
    contact_forces = torch.zeros(1, 4, 6)
    contact_forces[0, :, 0] = torch.tensor([14.9, 15.0, -20.0, 30.0])

    foot_contact_state = simulator._get_foot_contact_state(
        contact_geom_ids,
        contact_forces,
    )

    assert foot_contact_state["foot_ground_contact"].tolist() == [[True]]
    torch.testing.assert_close(
        foot_contact_state["foot_contact_normal_force"],
        torch.tensor([[35.0]]),
    )


def test_mujoco_foot_ground_contact_excludes_zero_force_at_zero_threshold(
    runtime_context,
    model_context,
):
    simulator = MujocoSimulator(runtime_context)
    simulator.model_context = model_context
    simulator.foot_contact_force_threshold = 0.0
    contact_geom_ids = torch.tensor([
        [[3, 0], [0, 3]],
        [[3, 0], [0, 3]],
    ])
    contact_forces = torch.zeros(2, 2, 6)
    contact_forces[0, 1, 0] = 2.0

    foot_contact_state = simulator._get_foot_contact_state(
        contact_geom_ids, contact_forces,
    )

    assert foot_contact_state["foot_ground_contact"].tolist() == [
        [True], [False],
    ]
    torch.testing.assert_close(
        foot_contact_state["foot_contact_normal_force"],
        torch.tensor([[2.0], [0.0]]),
    )


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


def test_mujoco_step_uses_native_nstep(runtime_context, monkeypatch):
    simulator = _configure_go1_simulator(
        runtime_context,
        num_envs=2,
        frame_skip=7,
    )
    calls = []

    def record_step(model, data, *, nstep):
        calls.append((model, data, nstep))

    monkeypatch.setattr(mujoco, "mj_step", record_step)

    try:
        simulator.step(torch.zeros(2, simulator.model_context.nu))
    finally:
        simulator.close()

    assert calls == [
        (simulator.models[0], simulator.datas[0], 7),
        (simulator.models[1], simulator.datas[1], 7),
    ]


def test_mujoco_step_reuses_persistent_executor(runtime_context):
    simulator = _configure_go1_simulator(
        runtime_context,
        num_envs=2,
        step_workers=2,
    )
    executor = simulator._step_executor

    try:
        action = torch.zeros(2, simulator.model_context.nu)
        simulator.step(action)
        simulator.step(action)
        assert simulator._step_executor is executor
    finally:
        simulator.close()

    assert executor is not None
    assert simulator._step_executor is None


def test_mujoco_rejects_invalid_step_workers(runtime_context):
    simulator = MujocoSimulator(runtime_context)

    with pytest.raises(ValueError, match="step_workers"):
        simulator.config_update(
            component=Component(None, None, None, None, None, None),
            step_workers=0,
        )


def test_mujoco_parallel_step_matches_serial_trajectory(runtime_context):
    serial = _configure_go1_simulator(
        runtime_context,
        num_envs=2,
        frame_skip=10,
    )
    parallel = _configure_go1_simulator(
        runtime_context,
        num_envs=2,
        frame_skip=10,
        step_workers=2,
    )

    try:
        serial.reset()
        parallel.reset()
        action = torch.linspace(
            -0.2,
            0.2,
            steps=2 * serial.model_context.nu,
        ).reshape(2, serial.model_context.nu)
        for _ in range(20):
            serial.step(action)
            parallel.step(action)

        for serial_data, parallel_data in zip(serial.datas, parallel.datas):
            np.testing.assert_array_equal(serial_data.qpos, parallel_data.qpos)
            np.testing.assert_array_equal(serial_data.qvel, parallel_data.qvel)
    finally:
        serial.close()
        parallel.close()


def test_mujoco_reuses_fixed_state_buffers(runtime_context):
    simulator = _configure_go1_simulator(runtime_context, num_envs=2)
    fixed_fields = (
        "qpos",
        "qvel",
        "qacc",
        "ctrl",
        "geom_xpos",
        "actuator_force",
        "base_lin_vel_body",
        "base_ang_vel_body",
        "geom_xvel",
    )

    try:
        simulator.reset()
        first = simulator.get_state()
        pointers = {
            name: getattr(first, name).data_ptr()
            for name in fixed_fields
        }
        second = simulator.get_state()

        assert {
            name: getattr(second, name).data_ptr()
            for name in fixed_fields
        } == pointers

        subset = simulator.get_state(torch.tensor([0]))
        assert all(
            getattr(subset, name).data_ptr() != pointers[name]
            for name in fixed_fields
        )
        assert all(
            getattr(subset, name).shape[0] == 1
            for name in fixed_fields
        )
        assert all(
            getattr(simulator._reset_state_numpy_buffers, name).shape[0] == 2
            for name in fixed_fields
        )

        larger_subset = simulator.get_state(torch.tensor([0, 1]))
        assert all(
            getattr(larger_subset, name).data_ptr()
            == getattr(subset, name).data_ptr()
            for name in fixed_fields
        )
        assert all(
            getattr(larger_subset, name).shape[0] == 2
            for name in fixed_fields
        )
    finally:
        simulator.close()
