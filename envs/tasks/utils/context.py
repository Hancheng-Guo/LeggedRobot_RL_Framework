import torch
from dataclasses import dataclass

from envs.simulators.utils.state import SimulatorState


@dataclass
class TaskContext:
    state: SimulatorState
    command: dict[str, torch.Tensor]
    action: torch.Tensor
    last_action: torch.Tensor
    episode_step: torch.Tensor
    step_dt: float
    env_ids: torch.Tensor | None = None


@dataclass
class TaskStepResult:
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
