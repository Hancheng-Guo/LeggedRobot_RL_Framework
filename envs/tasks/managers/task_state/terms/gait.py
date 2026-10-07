import math
import torch
from collections.abc import Sequence

from .base import BaseTaskStateTerm
from .registry import register_task_state
from envs.tasks.utils import TaskContext


_IDLE_SPEED_THRESHOLD = 0.1


@register_task_state
class QuadrupedalGaitPhase(BaseTaskStateTerm):

    def __init__(
        self,
        command_names: Sequence[str] = ("lin_vel_x", "lin_vel_y", "ang_vel_z"),
        omega_min: float = 4.0,
        omega_max: float = 10.0,
        command_gain: float = 1.0,
        *arg, **kawargs,
    ) -> None:
        
        super().__init__(*arg, **kawargs)

        if not math.isfinite(omega_min) or omega_min <= 0.0:
            raise ValueError("'omega_min' must be positive and finite.")
        if not math.isfinite(omega_max) or omega_max < omega_min:
            raise ValueError("'omega_max' must be finite and at least 'omega_min'.")
        if not math.isfinite(command_gain) or command_gain < 0.0:
            raise ValueError("'command_gain' must be non-negative and finite.")

        self._value = torch.zeros(
            (self.num_envs, 2),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.command_names = tuple(command_names)
        if not self.command_names:
            raise ValueError("'command_names' must not be empty.")
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.command_gain = command_gain


    @property
    def value(self) -> torch.Tensor:
        return self._value


    def update(
        self,
        task_context: TaskContext
    ) -> None:
        
        missing = set(self.command_names) - task_context.command.keys()
        if missing:
            raise ValueError(f"Unknown command name(s): {sorted(missing)}.")
        
        env_ids = task_context.env_ids

        command_norm = torch.linalg.norm(
            torch.cat(
                [task_context.command[name] for name in self.command_names],
                dim=-1,
            ),
            dim=-1
        )
        omega = (
            (self.omega_max - self.omega_min)
            * torch.tanh(self.command_gain * command_norm)
            + self.omega_min
        )
        d_phi = omega * task_context.step_dt
        moving = command_norm >= _IDLE_SPEED_THRESHOLD

        phase = self.value if env_ids is None else self.value[env_ids]
        moving_0 = phase[:, 0] + d_phi
        moving_1 = torch.where(
            moving_0 - phase[:, 1] > torch.pi,
            moving_0 - torch.pi,
            phase[:, 1],
        )
        idle_target = torch.ceil(phase / (2 * torch.pi)) * 2 * torch.pi
        idle = torch.minimum(phase + d_phi.unsqueeze(-1), idle_target)
        updated = torch.where(
            moving.unsqueeze(-1),
            torch.stack((moving_0, moving_1), dim=-1),
            idle,
        )
        if env_ids is None:
            self.value.copy_(updated)
        else:
            self.value[env_ids] = updated


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        if env_ids is None:
            self.value.zero_()
        else:
            self.value[env_ids] = 0.0
