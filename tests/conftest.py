from pathlib import Path

import pytest
import torch

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext


@pytest.fixture
def runtime_context() -> RuntimeContext:
    return RuntimeContext(
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_threads=1,
        seed=0,
        deterministic=True,
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
        base_pos_qpos_ids=torch.tensor([5, 6, 7]),
        base_quat_qpos_ids=torch.tensor([1, 2, 3, 4]),
        base_ang_vel_qvel_ids=torch.tensor([1, 2, 3]),
        body_names=("world", "base", "thigh", "foot"),
        gravity=torch.tensor([0.0, 0.0, -9.81]),
        joint_qpos_ids=torch.tensor([0, 8]),
        joint_qvel_ids=torch.tensor([0, 7]),
        actuator_ctrl_range=torch.tensor(
            [[0.0, 2.0], [-2.0, 2.0]],
            dtype=torch.float32,
        ),
        geom_names=("floor", "base", "thigh", "foot"),
        geom_body_ids=torch.tensor([0, 1, 2, 3]),
    )
