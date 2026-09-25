import torch
from dataclasses import dataclass

from envs.simulators.utils.state import SimulatorState


@dataclass
class TaskContext:
    """Built before command update for reward, then rebuilt after it for observation."""

    state: SimulatorState                   # State after the latest simulation step (or reset).
    command: dict[str, torch.Tensor]        # Command at context creation.
    last_command: dict[str, torch.Tensor]   # Command before the latest command update.
    action: torch.Tensor                    # Action just applied; zero after reset.
    last_action: torch.Tensor               # Action before the latest applied action.
    episode_step: torch.Tensor              # Completed control steps in each episode.
    step_dt: float                          # Duration of one control step.
    env_ids: torch.Tensor | None = None     # Selected environments; None means all.


@dataclass
class TaskStepResult:
    reward: torch.Tensor        # Reward for the completed step, scaled by step_dt.
    terminated: torch.Tensor    # Task termination on the completed step.
    truncated: torch.Tensor     # Time-limit truncation on the completed step.
