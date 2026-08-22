import torch
from dataclasses import dataclass
from typing import Any


@dataclass
class TaskContext:
    state: dict[str, torch.Tensor]
    command: dict[str, torch.Tensor]
    action: torch.Tensor
    last_action: torch.Tensor
    episode_step: torch.Tensor


@dataclass
class TaskStepResult:
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
