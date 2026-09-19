from pathlib import Path

import pytest
import torch

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext


BACKEND_MARKERS = {"mujoco", "isaacsim"}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "core: backend-independent unit tests")
    config.addinivalue_line("markers", "mujoco: MuJoCo backend tests")
    config.addinivalue_line("markers", "isaacsim: Isaac Sim backend tests")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Treat every test without a backend marker as a core test."""
    for item in items:
        if not any(item.get_closest_marker(name) for name in BACKEND_MARKERS):
            item.add_marker(pytest.mark.core)


@pytest.fixture
def runtime_context() -> RuntimeContext:
    return RuntimeContext(
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_threads=1,
        seed=0,
        deterministic_ops=True,
        load_dir=Path("."),
        save_dir=Path("."),
    )


@pytest.fixture
def model_context() -> ModelContext:
    return ModelContext(
        nq=9,
        nv=8,
        nu=2,
        na=0,
        base_id=1,
        base_pos_qpos_ids=torch.tensor([5, 6, 7]),
        base_quat_qpos_ids=torch.tensor([1, 2, 3, 4]),
        base_lin_vel_qvel_ids=torch.tensor([4, 5, 6]),
        base_ang_vel_qvel_ids=torch.tensor([1, 2, 3]),
        body_names=("world", "base", "thigh", "foot"),
        gravity=torch.tensor([0.0, 0.0, -9.81]),
        joint_qpos_ids=torch.tensor([0, 8]),
        joint_qvel_ids=torch.tensor([0, 7]),
        joint_default_pos=torch.tensor([0.0, 0.0]),
        joint_pos_limits=torch.tensor(
            [[-1.0, 1.0], [-2.0, 2.0]],
            dtype=torch.float32,
        ),
        actuator_ctrl_range=torch.tensor(
            [[0.0, 2.0], [-2.0, 2.0]],
            dtype=torch.float32,
        ),
        actuator_default_ctrl=torch.tensor([0.5, -0.5]),
        geom_names=("floor", "base", "thigh", "foot"),
        geom_body_ids=torch.tensor([0, 1, 2, 3]),
        foot_geom_ids=torch.tensor([3]),
        floor_geom_ids=torch.tensor([0]),
    )
