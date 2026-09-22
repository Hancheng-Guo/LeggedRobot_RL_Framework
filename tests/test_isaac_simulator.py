from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch

from envs.simulators.isaac_sim import IsaacSimSimulator
from envs.simulators.isaac_sim_backend import (
    IsaacSimModelMetadata,
    IsaacSimResetState,
)
from envs.simulators.registry import SIM_TYPE_MAP
from utils.component import Component


pytestmark = pytest.mark.isaacsim


class FakeIsaacBackend:
    def __init__(self, context) -> None:
        self.context = context
        self.configure_arguments: dict[str, object] = {}
        self.reset_ids: torch.Tensor | None = None
        self.step_action: torch.Tensor | None = None
        self.step_frame_skip: int | None = None
        self.closed = False

    def configure(self, **configuration: Any) -> None:
        self.configure_arguments = configuration
        self.num_envs = int(configuration["num_envs"])
        self.metadata = IsaacSimModelMetadata(
            body_names=("trunk", "FR_foot", "FL_foot"),
            dof_names=("FR_joint", "FL_joint"),
            joint_default_pos=torch.tensor((0.25, -0.25)),
            joint_pos_limits=torch.tensor(((-1.0, 1.0), (-2.0, 2.0))),
            actuator_ctrl_range=torch.tensor(((-1.0, 1.0), (-2.0, 2.0))),
            actuator_default_ctrl=torch.tensor((0.25, -0.25)),
            gravity=torch.tensor((0.0, 0.0, -9.81)),
            body_prim_paths=("/trunk", "/FR_foot", "/FL_foot"),
            base_body_prim_path="/trunk",
            foot_body_prim_paths=("/FR_foot", "/FL_foot"),
            floor_prim_paths=(
                tuple(configuration["floor_prim_paths"])
                or ("/World/GroundPlane",)
            ),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.reset_ids = env_ids

    def step(self, action: torch.Tensor, frame_skip: int) -> None:
        self.step_action = action
        self.step_frame_skip = frame_skip

    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        count = self.num_envs if env_ids is None else env_ids.numel()
        return {
            "qpos": torch.zeros(count, 9),
            "qvel": torch.zeros(count, 8),
            "qacc": torch.zeros(count, 8),
            "ctrl": torch.zeros(count, 2),
            "geom_xpos": torch.zeros(count, 4, 3),
            "geom_xvel": torch.zeros(count, 4, 6),
            "actuator_force": torch.zeros(count, 2),
            "base_lin_vel_body": torch.zeros(count, 3),
            "base_ang_vel_body": torch.zeros(count, 3),
            "contact_geom_ids": torch.full((count, 3, 2), -1),
            "contact_forces": torch.zeros(count, 3, 6),
            "foot_ground_contact": torch.zeros(
                count,
                3,
                2,
                dtype=torch.bool,
            ),
        }

    def render(self) -> np.ndarray:
        return np.zeros((4, 6, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _component() -> Component:
    return Component(None, None, None, None, None, None)


def _configure(
    tmp_path: Path,
    runtime_context,
    floor_prim_paths: tuple[str, ...] | None = None,
) -> tuple[IsaacSimSimulator, FakeIsaacBackend]:
    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot name='test'/>", encoding="utf-8")
    backend: FakeIsaacBackend | None = None

    def factory(context) -> FakeIsaacBackend:
        nonlocal backend
        backend = FakeIsaacBackend(context)
        return backend

    simulator = IsaacSimSimulator(runtime_context, backend_factory=factory)
    simulator.config_update(
        component=_component(),
        num_envs=2,
        model_path=model_path,
        sim_dt=0.002,
        frame_skip=10,
        render_mode="rgb_array",
        floor_prim_paths=floor_prim_paths,
    )
    assert backend is not None
    return simulator, backend


def test_isaac_sim_is_registered_without_importing_optional_runtime() -> None:
    assert SIM_TYPE_MAP["isaac_sim"] is IsaacSimSimulator


def test_isaac_sim_parses_reset_state() -> None:
    reset_state = IsaacSimSimulator._parse_reset_state({
        "base_position": [0.0, 0.0, 0.27],
        "base_orientation": [1.0, 0.0, 0.0, 0.0],
        "joint_positions": {"FR_hip_joint": 0.1},
    })

    assert reset_state == IsaacSimResetState(
        base_position=(0.0, 0.0, 0.27),
        base_orientation=(1.0, 0.0, 0.0, 0.0),
        joint_positions={"FR_hip_joint": 0.1},
    )


def test_isaac_sim_rejects_unknown_reset_state_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported reset state field"):
        IsaacSimSimulator._parse_reset_state({"unknown": 1.0})


def test_simulator_registry_does_not_eagerly_import_backends() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import envs.simulators.registry; "
                "assert 'envs.simulators.mujoco' not in sys.modules; "
                "assert 'envs.simulators.isaac_sim' not in sys.modules"
            ),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_isaac_sim_builds_framework_model_context(
    tmp_path: Path,
    runtime_context,
) -> None:
    simulator, backend = _configure(tmp_path, runtime_context)

    context = simulator.model_context
    assert context.nq == 9
    assert context.nv == 8
    assert context.nu == 2
    assert context.base_id == 0
    assert context.base_pos_qpos_ids.tolist() == [0, 1, 2]
    assert context.base_quat_qpos_ids.tolist() == [3, 4, 5, 6]
    assert context.joint_qpos_ids.tolist() == [7, 8]
    assert context.joint_qvel_ids.tolist() == [6, 7]
    assert context.geom_names == (
        "trunk",
        "FR_foot",
        "FL_foot",
        "GroundPlane",
    )
    assert context.foot_geom_ids.tolist() == [1, 2]
    assert context.floor_geom_ids.tolist() == [3]
    assert backend.configure_arguments["model_path"] == (
        tmp_path / "robot.urdf"
    ).resolve()
    assert backend.configure_arguments["floor_prim_paths"] == ()


def test_isaac_sim_forwards_reset_step_state_render_and_close(
    tmp_path: Path,
    runtime_context,
) -> None:
    simulator, backend = _configure(tmp_path, runtime_context)
    env_ids = torch.tensor((1,))
    action = torch.zeros(2, 2)

    simulator.reset(env_ids)
    simulator.step(action)
    state = simulator.get_state(env_ids)
    frame = simulator.render()
    simulator.close()

    assert backend.reset_ids is env_ids
    assert backend.step_action is action
    assert backend.step_frame_skip == 10
    assert state.qpos.shape == (1, 9)
    assert frame is not None
    assert frame.shape == (4, 6, 3)
    assert backend.closed


def test_isaac_sim_reuses_backend_when_stage_configuration_is_unchanged(
    tmp_path: Path,
    runtime_context,
) -> None:
    simulator, backend = _configure(tmp_path, runtime_context)

    simulator.config_update(
        component=_component(),
        num_envs=2,
    )

    assert simulator._backend is backend
    assert not backend.closed


def test_isaac_sim_rebuilds_backend_when_environment_count_changes(
    tmp_path: Path,
    runtime_context,
) -> None:
    simulator, original_backend = _configure(tmp_path, runtime_context)

    simulator.config_update(
        component=_component(),
        num_envs=3,
    )

    assert simulator._backend is not original_backend
    assert original_backend.closed
    assert simulator.num_envs == 3


def test_isaac_sim_builds_multiple_floor_geom_ids(
    tmp_path: Path,
    runtime_context,
) -> None:
    simulator, _ = _configure(
        tmp_path,
        runtime_context,
        floor_prim_paths=("/World/Floor", "/World/Ramp"),
    )

    assert simulator.model_context.geom_names[-2:] == ("Floor", "Ramp")
    assert simulator.model_context.floor_geom_ids.tolist() == [3, 4]
    assert simulator.model_context.geom_body_ids.tolist() == [0, 1, 2, -1, -1]


def test_isaac_sim_rejects_unsupported_model_formats(
    tmp_path: Path,
    runtime_context,
) -> None:
    model_path = tmp_path / "robot.sdf"
    model_path.write_text("<sdf/>", encoding="utf-8")
    simulator = IsaacSimSimulator(
        runtime_context,
        backend_factory=FakeIsaacBackend,
    )

    with pytest.raises(ValueError, match="requires a USD model"):
        simulator.config_update(
            component=_component(),
            num_envs=1,
            model_path=model_path,
            sim_dt=0.002,
            frame_skip=1,
        )


def test_isaac_sim_validates_action_shape(
    tmp_path: Path,
    runtime_context,
) -> None:
    simulator, _ = _configure(tmp_path, runtime_context)

    with pytest.raises(ValueError, match="Action must have shape"):
        simulator.step(torch.zeros(2, 3))
