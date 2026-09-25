import math

import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


def _foot_contact_output_dim(model_context: ModelContext) -> int:
    if model_context.foot_geom_ids.numel() == 0:
        raise ValueError(
            "Foot contact observations require at least one foot geom."
        )
    if model_context.floor_geom_ids.numel() == 0:
        raise ValueError(
            "Foot contact observations require at least one floor geom."
        )
    return model_context.foot_geom_ids.numel()


@register_observation
class FootHeight(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if model_context.foot_geom_ids.numel() == 0:
            raise ValueError(
                "FootHeight requires at least one foot geom."
            )

        self.foot_geom_ids = model_context.foot_geom_ids
        self.output_dim = self.foot_geom_ids.numel()


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        
        return task_context.state.geom_xpos[:, self.foot_geom_ids, 2]



@register_observation
class FootContactNormalForce(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.output_dim = _foot_contact_output_dim(model_context)


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        
        return task_context.state.foot_contact_normal_force


@register_observation
class FootContactState(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.output_dim = _foot_contact_output_dim(model_context)


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        return task_context.state.foot_ground_contact.to(self.context.dtype)


@register_observation
class FootDurationTanh(BaseObservationTerm):

    def __init__(
        self,
        num_envs: int,
        model_context: ModelContext,
        alpha: float = 1.0,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not math.isfinite(alpha)
            or alpha <= 0.0
        ):
            raise ValueError("'alpha' must be a finite positive number.")

        self.alpha = float(alpha)
        self.foot_geom_ids = model_context.foot_geom_ids
        self.last_foot_state = torch.zeros(
            (num_envs, self.foot_geom_ids.numel()),
            dtype=torch.bool,
            device=self.context.device,
        )
        self.initialized = torch.zeros_like(self.last_foot_state)
        self.duration = torch.zeros(
            (num_envs, self.foot_geom_ids.numel()),
            dtype=self.context.dtype,
            device=self.context.device,
        )

        self.output_dim = _foot_contact_output_dim(model_context)


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        
        landed = task_context.state.foot_ground_contact
        env_ids = (
            slice(None)
            if task_context.env_ids is None
            else task_context.env_ids
        )

        last_foot_state = self.last_foot_state[env_ids]
        initialized = self.initialized[env_ids]
        duration = self.duration[env_ids]
        unchanged = initialized & (landed == last_foot_state)
        duration = torch.where(
            unchanged,
            duration + task_context.step_dt,
            torch.zeros_like(duration),
        )
        self.duration[env_ids] = duration
        self.last_foot_state[env_ids] = landed
        self.initialized[env_ids] = True
        sign = torch.where(
            landed,
            torch.ones_like(duration),
            -torch.ones_like(duration),
        )

        return torch.tanh(duration * self.alpha) * sign


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self.last_foot_state.zero_()
            self.initialized.zero_()
            self.duration.zero_()
        else:
            self.last_foot_state[env_ids] = False
            self.initialized[env_ids] = False
            self.duration[env_ids] = 0.0
