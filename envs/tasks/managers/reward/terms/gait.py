import torch
from collections.abc import Sequence

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.managers.reward.terms.utils import command_vector
from envs.tasks.utils.context import TaskContext


_FOOT_PHASE_REAL_NAME = "foot_phase_real"
_FOOT_PHASE_IMAG_NAME = "foot_phase_imag"
_HALF_PERIOD_DURATION_NAME = "half_period_duration"

_LANDED_DISTANCE_FACTOR = -1.0
_LIFTED_DISTANCE_FACTOR = -0.25

_IDLE_SPEED_THRESHOLD = 1.0e-7
_TROT_LOOPS = {
    False: ((0b1111, 0),),
    True: (
        (0b1111, 2),
        (0b1011, 1),
        (0b1001, 4),
        (0b1101, 3),
        (0b1111, 2),
        (0b0111, 1),
        (0b0110, 4),
        (0b1110, 3),
    ),
}


def _named_tensor_squeeze(
    task_context: TaskContext,
    name: str,
) -> torch.Tensor:
    
    if name in task_context.command:
        return task_context.command[name].squeeze(-1)
    if hasattr(task_context.state, name):
        return getattr(task_context.state, name).squeeze(-1)
    raise ValueError(f"'{name}' is missing from task context.")


@register_reward
class TrotLoopTanh(BaseRewardTerm):

    def __init__(
        self,
        num_envs: int,
        model_context: ModelContext,
        command_names: Sequence[str] = (
            "lin_vel_x",
            "lin_vel_y",
            "ang_vel_z",
        ),
        growth_rate: float = 1.0,
        min_phase_duration: float = 0.04,
        max_phase_duration: float = 0.30,
        phase_duration_std: float = 0.10,
        early_transition_penalty: float = 1.0,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if model_context.foot_geom_ids.numel() != 4:
            raise ValueError(
                "TrotLoopTanh requires exactly four foot geoms."
            )
        if not command_names or any(
            not isinstance(name, str) or not name
            for name in command_names
        ):
            raise ValueError("'command_names' must contain valid names.")
        if min_phase_duration < 0.0:
            raise ValueError("'min_phase_duration' must be non-negative.")
        if max_phase_duration <= min_phase_duration:
            raise ValueError(
                "'max_phase_duration' must be greater than "
                "'min_phase_duration'."
            )
        if phase_duration_std <= 0.0:
            raise ValueError("'phase_duration_std' must be positive.")
        if early_transition_penalty < 0.0:
            raise ValueError("'early_transition_penalty' must be non-negative.")

        self.command_names = tuple(command_names)
        self.growth_rate = growth_rate
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        self.phase_duration_std = phase_duration_std
        self.early_transition_penalty = early_transition_penalty
        self.gait_is_moving: list[bool | None] = [None] * num_envs
        self.gait_phases: list[list[int]] = [
            [] for _ in range(num_envs)
        ]
        self.foot_states: list[int | None] = [None] * num_envs
        self.gait_loop_duration = torch.zeros(
            num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.phase_duration = torch.zeros_like(self.gait_loop_duration)


    @staticmethod
    def _initial_phases(
        loop: tuple[tuple[int, int], ...],
        foot_state: int,
    ) -> list[int]:
        
        return [
            phase
            for phase, (state, _) in enumerate(loop)
            if foot_state == state
        ]


    @staticmethod
    def _advanced_phases(
        loop: tuple[tuple[int, int], ...],
        phases: list[int],
        foot_state: int,
    ) -> list[int]:
        
        next_phases: list[int] = []
        for phase in phases:
            max_advance = loop[phase][1]
            for advance in range(1, max_advance + 1):
                candidate = (phase + advance) % len(loop)
                if loop[candidate][0] == foot_state:
                    next_phases.append(candidate)
                    break

        return next_phases


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        
        foot_contact = task_context.state.foot_ground_contact.any(dim=1)
        foot_states = (
            foot_contact[:, 0].long() * 0b1000
            + foot_contact[:, 1].long() * 0b0100
            + foot_contact[:, 2].long() * 0b0010
            + foot_contact[:, 3].long()
        ).cpu().tolist()
        speed = torch.linalg.vector_norm(
            command_vector(task_context, self.command_names),
            dim=-1,
        )
        moving = (speed >= _IDLE_SPEED_THRESHOLD).detach().cpu().tolist()
        early_transition_penalty = torch.zeros_like(self.phase_duration)

        for env_id, (is_moving, foot_state) in enumerate(
            zip(moving, foot_states)
        ):
            loop = _TROT_LOOPS[is_moving]
            phases = self.gait_phases[env_id]
            same_mode = is_moving == self.gait_is_moving[env_id]
            same_state = foot_state == self.foot_states[env_id]

            if not same_mode or not phases:
                self.gait_is_moving[env_id] = is_moving
                phases = self._initial_phases(loop, foot_state)
                self.gait_loop_duration[env_id] = (
                    task_context.step_dt if phases else 0.0
                )
                self.phase_duration[env_id] = task_context.step_dt
            elif same_state:
                self.gait_loop_duration[env_id] += task_context.step_dt
                self.phase_duration[env_id] += task_context.step_dt
            else:
                next_phases = self._advanced_phases(loop, phases, foot_state)
                if next_phases:
                    if (
                        is_moving
                        and self.phase_duration[env_id] < self.min_phase_duration
                    ):
                        early_transition_penalty[env_id] = (
                            self.early_transition_penalty
                            * (
                                self.min_phase_duration
                                - self.phase_duration[env_id]
                            )
                            / max(
                                self.min_phase_duration,
                                torch.finfo(self.context.dtype).eps,
                            )
                        )
                    phases = next_phases
                    self.gait_loop_duration[env_id] += task_context.step_dt
                    self.phase_duration[env_id] = task_context.step_dt
                else:
                    phases = self._initial_phases(loop, foot_state)
                    self.gait_loop_duration[env_id] = 0.0
                    self.phase_duration[env_id] = task_context.step_dt

            self.gait_phases[env_id] = phases
            self.foot_states[env_id] = foot_state

        overtime = (
            self.phase_duration - self.max_phase_duration
        ).clamp_min(0.0)
        timeout_decay = torch.exp(
            -(overtime / self.phase_duration_std).square()
        )
        timeout_decay = torch.where(
            torch.as_tensor(moving, device=self.context.device),
            timeout_decay,
            torch.ones_like(timeout_decay),
        )

        return timeout_decay * torch.tanh(
            self.growth_rate * self.gait_loop_duration
        ) - early_transition_penalty


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self.gait_loop_duration.zero_()
            self.phase_duration.zero_()
            self.gait_is_moving = [None] * len(self.gait_is_moving)
            self.gait_phases = [[] for _ in self.gait_phases]
            self.foot_states = [None] * len(self.foot_states)
            return

        self.gait_loop_duration[env_ids] = 0.0
        self.phase_duration[env_ids] = 0.0
        for env_id in env_ids.cpu().tolist():
            self.gait_is_moving[env_id] = None
            self.gait_phases[env_id] = []
            self.foot_states[env_id] = None


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

        self.foot_geom_ids = model_context.foot_geom_ids
        if self.foot_geom_ids.numel() != 4:
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
        target_phase = torch.pi * torch.sin(
            torch.pi * target_phase_steps / half_period_steps.unsqueeze(-1)
        )

        foot_height = self._foot_height_from_base_plane(task_context)
        foot_phase = torch.pi * torch.cos(
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
        
        phase_real = _named_tensor_squeeze(
            task_context,
            _FOOT_PHASE_REAL_NAME
        )
        phase_imag = _named_tensor_squeeze(
            task_context,
            _FOOT_PHASE_IMAG_NAME
        )
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
        
        half_period_duration = _named_tensor_squeeze(
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

        qpos = task_context.state.qpos
        base_pos = qpos[:, self.base_pos_qpos_ids]
        quaternion = qpos[:, self.base_quat_qpos_ids]
        quaternion = quaternion / quaternion.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(quaternion.dtype).eps)

        foot_pos = task_context.state.geom_xpos[:, self.foot_geom_ids, :]

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
