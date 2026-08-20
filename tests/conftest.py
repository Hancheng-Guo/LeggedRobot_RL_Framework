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
        actuator_ctrl_range=torch.tensor(
            [[0.0, 2.0], [-2.0, 2.0]],
            dtype=torch.float32,
        ),
    )

