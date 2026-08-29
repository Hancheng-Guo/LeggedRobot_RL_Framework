import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.managers.reward.terms.utils import geom_ids_from_names
from envs.tasks.utils.context import TaskContext


@register_reward
class IllegalContactL1(BaseRewardTerm):

    def __init__(
        self,
        model_context: ModelContext,
        num_envs: int,
        geom_legal_names: list[str] | None = None,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if model_context.geom_floor_ids.numel() == 0:
            raise ValueError(
                "IllegalContactL1 requires 'model_context.geom_floor_ids'."
            )
        self.num_envs = num_envs
        self.legal_geom_ids = geom_ids_from_names(
            model_context,
            geom_legal_names or [],
            fallback_ids=model_context.geom_foot_ids,
        )
        self.ground_geom_ids = model_context.geom_floor_ids


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        contact_geom_ids = task_context.state["contact_geom_ids"]
        if contact_geom_ids.shape[1] == 0:
            return torch.zeros(
                self.num_envs,
                dtype=self.context.dtype,
                device=self.context.device,
            )

        geom1 = contact_geom_ids[..., 0]
        geom2 = contact_geom_ids[..., 1]
        illegal_ground = (
            (~torch.isin(geom1, self.legal_geom_ids) & torch.isin(geom2, self.ground_geom_ids))
            | (torch.isin(geom1, self.ground_geom_ids) & ~torch.isin(geom2, self.legal_geom_ids))
        )
        return illegal_ground.sum(dim=-1).to(dtype=self.context.dtype)
