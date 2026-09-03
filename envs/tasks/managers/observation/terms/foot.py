import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


@register_observation
class FootHeight(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if model_context.geom_foot_ids.numel() == 0:
            raise ValueError(
                "FootHeight requires at least one foot geom."
            )

        self.geom_foot_ids = model_context.geom_foot_ids
        self.output_dim = self.geom_foot_ids.numel()


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        return task_context.state["geom_xpos"][:, self.geom_foot_ids, 2]


class _FootContactObservation(BaseObservationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if model_context.geom_foot_ids.numel() == 0:
            raise ValueError(
                "Foot contact observations require at least one foot geom."
            )
        if model_context.geom_floor_ids.numel() == 0:
            raise ValueError(
                "Foot contact observations require at least one floor geom."
            )

        self.geom_foot_ids = model_context.geom_foot_ids
        self.geom_floor_ids = model_context.geom_floor_ids
        self.output_dim = self.geom_foot_ids.numel()


    def _normal_forces(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        contact_geom_ids = task_context.state["contact_geom_ids"]
        contact_forces = task_context.state["contact_forces"]

        if contact_geom_ids.ndim != 3 or contact_geom_ids.shape[-1] != 2:
            raise ValueError(
                "'contact_geom_ids' must have shape "
                "[num_envs, num_contacts, 2]."
            )
        if (
            contact_forces.ndim != 3
            or contact_forces.shape[:2] != contact_geom_ids.shape[:2]
            or contact_forces.shape[-1] < 1
        ):
            raise ValueError(
                "'contact_forces' must have shape "
                "[num_envs, num_contacts, at_least_1]."
            )

        geom1 = contact_geom_ids[..., 0]
        geom2 = contact_geom_ids[..., 1]
        normal_force = contact_forces[..., 0].abs()
        forces = torch.zeros(
            (contact_geom_ids.shape[0], self.output_dim),
            dtype=contact_forces.dtype,
            device=contact_forces.device,
        )

        for index, foot_geom_id in enumerate(self.geom_foot_ids):
            foot_ground_contact = (
                ((geom1 == foot_geom_id) & torch.isin(geom2, self.geom_floor_ids))
                | ((geom2 == foot_geom_id) & torch.isin(geom1, self.geom_floor_ids))
            )
            forces[:, index] = (
                normal_force * foot_ground_contact
            ).sum(dim=-1)

        return forces


@register_observation
class FootContactNormalForce(_FootContactObservation):

    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        return self._normal_forces(task_context)


@register_observation
class FootContactState(_FootContactObservation):

    def __init__(
        self,
        threshold: float = 15.0,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if threshold < 0.0:
            raise ValueError("'threshold' must be non-negative.")
        self.threshold = threshold


    def compute(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:
        return (
            self._normal_forces(task_context) >= self.threshold
        ).to(self.context.dtype)
