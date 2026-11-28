import math
import torch
from collections.abc import Sequence
from typing import Any

from .base import BaseObservationTerm
from .registry import register_observation
from envs.simulators.utils import ModelContext
from envs.tasks.utils import TaskContext, validate_height_reference, foot_heights


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
        height_reference: str = "ground",
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)
        self.height_reference = validate_height_reference(height_reference)

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
        
        return foot_heights(task_context, self.foot_geom_ids, self.height_reference)



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

    RESET_STATE_FIELDS = ('duration', 'initialized', 'last_foot_state')

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


class _FootTerrainObservation(BaseObservationTerm):

    field: str

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        shapes = model_context.state_term_output_shapes.get(
            "foot_terrain_scan", {}
        )
        if self.field not in shapes:
            raise ValueError(
                f"{self.name} requires simulator_state_manager_config."
                "terms.foot_terrain_scan."
            )
        self.output_dim = math.prod(shapes[self.field])


    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "field": self.field,
        }


    def playback_state_dict(
        self,
        env_index: int,
    ) -> dict[str, Any]:
        return self.checkpoint_state_dict()


@register_observation
class FootTerrainScanHeights(_FootTerrainObservation):

    field = "foot_terrain_scan_heights"

    def __init__(
        self,
        clip_range: Sequence[float] | None = None,
        *args, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clip_range = None
        if clip_range is not None:
            lower, upper = clip_range
            if not (
                math.isfinite(lower)
                and math.isfinite(upper)
                and lower <= 0 <= upper
            ):
                raise ValueError(
                    "foot_terrain_scan_heights.clip_range must be "
                    "finite and contain zero."
                )
            self.clip_range = (float(lower), float(upper))


    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "clip_range": self.clip_range,
            **super()._checkpoint_metadata(),
        }


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        heights = task_context.state.foot_terrain_scan_heights
        assert heights is not None
        heights = heights.flatten(1)
        if self.clip_range is not None:
            heights = heights.clamp(*self.clip_range)
        return heights


@register_observation
class FootTerrainScanValid(_FootTerrainObservation):

    field = "foot_terrain_scan_valid"

    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        valid = task_context.state.foot_terrain_scan_valid
        assert valid is not None
        return valid.flatten(1).to(self.context.dtype)
