import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import register_observation
from envs.tasks.utils.context import TaskContext


def _foot_contact_output_dim(model_context: ModelContext) -> int:
    if model_context.geom_foot_ids.numel() == 0:
        raise ValueError(
            "Foot contact observations require at least one foot geom."
        )
    if model_context.geom_floor_ids.numel() == 0:
        raise ValueError(
            "Foot contact observations require at least one floor geom."
        )
    return model_context.geom_foot_ids.numel()


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
        
        contact_forces = task_context.state["contact_forces"]
        foot_ground_contact = task_context.state["foot_ground_contact"]

        if (
            contact_forces.ndim != 3
            or contact_forces.shape[-1] < 1
        ):
            raise ValueError(
                "'contact_forces' must have shape "
                "[num_envs, num_contacts, at_least_1]."
            )
        if (
            foot_ground_contact.ndim != 3
            or foot_ground_contact.shape[:2] != contact_forces.shape[:2]
            or foot_ground_contact.shape[-1] != self.output_dim
        ):
            raise ValueError(
                "'foot_ground_contact' must have shape "
                "[num_envs, num_contacts, num_feet]."
            )

        normal_force = contact_forces[..., 0].abs()
        return (
            normal_force.unsqueeze(-1) * foot_ground_contact
        ).sum(dim=1)


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
        return task_context.state["foot_ground_contact"].any(
            dim=1,
        ).to(self.context.dtype)
