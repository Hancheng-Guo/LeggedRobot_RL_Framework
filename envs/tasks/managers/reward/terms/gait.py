import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.utils.context import TaskContext


_FOOT_PHASE_REAL_NAME = "foot_phase_real"
_FOOT_PHASE_IMAG_NAME = "foot_phase_imag"
_HALF_PERIOD_DURATION_NAME = "half_period_duration"

_LANDED_DISTANCE_FACTOR = -1.0
_LIFTED_DISTANCE_FACTOR = -0.25


def _named_tensor(
    task_context: TaskContext,
    name: str,
) -> torch.Tensor:
    
    if name in task_context.command:
        return task_context.command[name]
    if name in task_context.state:
        return task_context.state[name]
    raise ValueError(f"'{name}' is missing from task context.")


@register_reward
class QuadrupedalGaitPhaseL2Exp(BaseRewardTerm):

    def __init__(
        self,
        num_envs: int,
        model_context: ModelContext,
        target_height: float,
        sigma: float = 1,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.geom_foot_ids = model_context.geom_foot_ids
        if self.geom_foot_ids.numel() != 4:
            raise ValueError(
                "QuadrupedalGaitPhaseL2Exp requires exactly four foot geoms."
            )
        if sigma <= 0.0:
            raise ValueError("'sigma' must be positive.")

        self.base_pos_qpos_ids = model_context.base_pos_qpos_ids
        self.base_quat_qpos_ids = model_context.base_quat_qpos_ids
        self.target_height = target_height
        self.sigma = sigma

        self.foot_phase_steps = torch.zeros(
            (num_envs, 4),
            dtype=self.context.dtype,
            device=self.context.device,
        )


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        half_period_durations = self._half_period_durations(task_context)
        half_period_steps = torch.round(
            half_period_durations / task_context.step_dt
        ).clamp_min(1.0)
        target_phase_steps = self._update_phase_steps(half_period_steps)
        target_phase = torch.sin(
            torch.pi * target_phase_steps / half_period_steps.unsqueeze(-1)
        )

        foot_height = self._foot_height_from_base_plane(task_context)
        foot_phase = torch.cos(
            torch.pi * (
                (foot_height - _LIFTED_DISTANCE_FACTOR * self.target_height) /
                (
                    self.target_height * 
                    (_LANDED_DISTANCE_FACTOR - _LIFTED_DISTANCE_FACTOR)
                )
            )
        )

        phase_shift_real, phase_shift_imag = (
            self._check_phase_inputs(task_context)
        )
        phase_shift = torch.atan2(phase_shift_imag, phase_shift_real)
        phase_error = torch.atan2(
            torch.sin(foot_phase + phase_shift - target_phase),
            torch.cos(foot_phase + phase_shift - target_phase),
        )

        return torch.mean(
            torch.exp(-(phase_error / self.sigma).square()),
            dim=-1,
        )


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        if env_ids is None:
            self.foot_phase_steps.zero_()
        else:
            self.foot_phase_steps[env_ids] = 0.0


    def _check_phase_inputs(
        self,
        task_context: TaskContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        phase_real = _named_tensor(task_context, _FOOT_PHASE_REAL_NAME)
        phase_imag = _named_tensor(task_context, _FOOT_PHASE_IMAG_NAME)
        if phase_real.shape != self.foot_phase_steps.shape:
            raise ValueError(
                f"'{_FOOT_PHASE_REAL_NAME}' must have shape "
                f"{tuple(self.foot_phase_steps.shape)}."
            )
        if phase_imag.shape != self.foot_phase_steps.shape:
            raise ValueError(
                f"'{_FOOT_PHASE_IMAG_NAME}' must have shape "
                f"{tuple(self.foot_phase_steps.shape)}."
            )

        return phase_real, phase_imag


    def _half_period_durations(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        half_period_duration = _named_tensor(
            task_context,
            _HALF_PERIOD_DURATION_NAME
        )
        if half_period_duration.ndim != 1:
            raise ValueError(
                f"'{_HALF_PERIOD_DURATION_NAME}' must have shape "
                f"({self.foot_phase_steps.shape[0]},)."
            )

        return half_period_duration.clamp_min(0.05)


    def _foot_height_from_base_plane(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:

        qpos = task_context.state["qpos"]
        base_pos = qpos[:, self.base_pos_qpos_ids]
        quaternion = qpos[:, self.base_quat_qpos_ids]
        quaternion = quaternion / quaternion.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(quaternion.dtype).eps)

        foot_pos = task_context.state["geom_xpos"][:, self.geom_foot_ids, :]

        w = quaternion[:, 0:1]
        xyz = quaternion[:, 1:4]
        local_z = quaternion.new_tensor([0.0, 0.0, 1.0]).expand_as(xyz)
        base_plane_normal = (
            local_z * (2.0 * w.square() - 1.0)
            + 2.0 * w * torch.cross(xyz, local_z, dim=-1)
            + 2.0 * xyz * (xyz * local_z).sum(dim=-1, keepdim=True)
        )

        return (
            (foot_pos - base_pos.unsqueeze(1))
            * base_plane_normal.unsqueeze(1)
        ).sum(dim=-1)


    def _update_phase_steps(
        self,
        half_period_steps: torch.Tensor
    ) -> torch.Tensor:

        fl_phase = self.foot_phase_steps[:, 0]
        fr_phase = self.foot_phase_steps[:, 1]
        phase_gap = fr_phase - fl_phase

        increment_all = phase_gap == half_period_steps
        increment_fr_rl = phase_gap < half_period_steps
        increment_fl_rr = phase_gap > half_period_steps

        self.foot_phase_steps[:, [0, 3]] += increment_fl_rr.unsqueeze(-1).to(
            self.foot_phase_steps.dtype
        )
        self.foot_phase_steps[:, [1, 2]] += increment_fr_rl.unsqueeze(-1).to(
            self.foot_phase_steps.dtype
        )
        self.foot_phase_steps += increment_all.unsqueeze(-1).to(
            self.foot_phase_steps.dtype
        )

        self.foot_phase_steps[:, 3] = self.foot_phase_steps[:, 0]
        self.foot_phase_steps[:, 2] = self.foot_phase_steps[:, 1]

        return self.foot_phase_steps
