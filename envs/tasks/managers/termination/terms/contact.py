import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.termination.terms.base import BaseTerminationTerm
from envs.tasks.managers.termination.terms.registry import register_termination
from envs.tasks.utils.context import TaskContext


@register_termination
class BodyContact(BaseTerminationTerm):

    def __init__(
        self,
        model_context: ModelContext,
        body_names: list[str],
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if model_context.geom_floor_ids.numel() == 0:
            raise ValueError(
                "BodyContact requires 'model_context.geom_floor_ids'."
            )
        self.geom_floor_ids = model_context.geom_floor_ids

        self.geom_body_ids = self._extract_geom_ids(
            model_context,
            body_names,
        )


    def _extract_geom_ids(
        self,
        model_context: ModelContext,
        body_names: list[str],
    ) -> torch.Tensor:
        
        if not body_names:
            raise ValueError("'body_names' must be a non-empty list.")

        # get body IDs from names
        body_name_to_id = {
            name: body_id
            for body_id, name in enumerate(model_context.body_names)
            if name is not None
        }
        unknown_body_names = set(body_names) - body_name_to_id.keys()
        if unknown_body_names:
            raise ValueError(
                f"Unknown body name(s): {sorted(unknown_body_names)}."
            )

        # get body IDs of the specified bodies discribed in 'body_names'
        body_ids = torch.tensor(
            [body_name_to_id[name] for name in body_names],
            dtype=torch.long,
            device=self.context.device,
        )

        # get IDs of all geoms for the specified bodies
        geom_ids = torch.nonzero(
            torch.isin(model_context.geom_body_ids, body_ids),
            as_tuple=False,
        ).squeeze(-1)
        if geom_ids.numel() == 0:
            raise ValueError(
                "The specified bodies do not contain any geoms."
            )

        return geom_ids


    def compute(
        self,
        task_context: TaskContext
    ) -> torch.Tensor:
        
        contact_geom_ids = task_context.state["contact_geom_ids"]
        if contact_geom_ids.ndim != 3 or contact_geom_ids.shape[-1] != 2:
            raise ValueError(
                "'contact_geom_ids' must have shape "
                "[num_envs, num_contacts, 2]."
            )

        geom1 = contact_geom_ids[..., 0]
        geom2 = contact_geom_ids[..., 1]
        target_is_geom1 = torch.isin(geom1, self.geom_body_ids)
        target_is_geom2 = torch.isin(geom2, self.geom_body_ids)
        ground_is_geom1 = torch.isin(geom1, self.geom_floor_ids)
        ground_is_geom2 = torch.isin(geom2, self.geom_floor_ids)

        return (
            (target_is_geom1 & ground_is_geom2)
            | (ground_is_geom1 & target_is_geom2)
        ).any(dim=1)
