"""Result of a completed task step."""
import torch
from dataclasses import dataclass


@dataclass
class TaskStepResult:
    reward: torch.Tensor        # Reward for the completed step, scaled by step_dt.
    terminated: torch.Tensor    # Task termination on the completed step.
    truncated: torch.Tensor     # Time-limit truncation on the completed step.
