import torch
from dataclasses import dataclass


@dataclass
class TaskContext:
    state: dict[str, torch.Tensor]
    command: dict[str, torch.Tensor]
    action: torch.Tensor
    last_action: torch.Tensor
    episode_step: torch.Tensor