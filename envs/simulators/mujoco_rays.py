"""Batched scene queries using the live per-environment MuJoCo geometry."""

import mujoco
import torch
import numpy as np
from copy import copy


class MujocoRayQuery:

    def __init__(
        self,
        models: list,
        datas: list,
        floor_ids: list[int],
        base_id: int,
    ) -> None:

        self.models = models
        self.datas = datas
        self.floor_ids = floor_ids

        ref_model = models[0]
        geom_groups = ref_model.geom_group
        body_ids = ref_model.geom_bodyid
        self.floor_groups = np.unique(geom_groups[floor_ids])

        # Identify the floor geoms and this robot's moving visual geoms.
        floor_mask = np.isin(np.arange(ref_model.ngeom), floor_ids)
        robot_root_id = ref_model.body_rootid[base_id]
        robot_mask = ref_model.body_rootid[body_ids] == robot_root_id
        moving_mask = ref_model.body_weldid[body_ids] != 0
        robot_visual_mask = (geom_groups == 2) & robot_mask & moving_mask & ~floor_mask

        floor_material_ids = ref_model.geom_matid[floor_ids]
        assigned_material_ids = floor_material_ids[floor_material_ids >= 0]
        hidden_floor = (
            np.any(ref_model.geom_rgba[floor_ids, 3] == 0)
            or np.any(ref_model.mat_rgba[assigned_material_ids, 3] == 0)
        )
        groups_overlap = (
            np.any(np.isin(geom_groups, self.floor_groups) != floor_mask)
            or np.any((geom_groups == 2) != robot_visual_mask)
        )

        # Isolate the queried geoms when the original grouping or visibility would affect hits.
        self.query_model = None
        if hidden_floor or groups_overlap:
            self.query_model = copy(ref_model)
            self.query_model.geom_group[:] = 5
            self.query_model.geom_group[floor_mask] = 0
            self.query_model.geom_group[robot_visual_mask] = 2
            # The floor participates through collision geometry, even when transparent.
            self.query_model.geom_matid[floor_mask] = -1
            self.query_model.geom_rgba[floor_mask, 3] = 1
            self.floor_groups = [0]


    def __call__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
        *,
        include_ground: bool,
        include_robot: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        groups = np.zeros(6, dtype=np.uint8)
        if include_ground:
            groups[self.floor_groups] = 1
        if include_robot:
            groups[2] = 1

        starts = origins.detach().to("cpu", torch.float64).numpy()
        vectors = directions.detach().to("cpu", torch.float64).contiguous().numpy()
        distances = np.full(vectors.shape[:-1], np.nan)
        ground_hits = np.zeros(distances.shape, dtype=bool)
        selected = range(len(self.models)) if env_ids is None else env_ids.tolist()
        for row, env_id in enumerate(selected):
            model = self.models[env_id]
            if self.query_model is not None:
                self.query_model.hfield_data[:] = model.hfield_data
                model = self.query_model
            for origin, rays, result, ground in zip(
                starts[row], vectors[row], distances[row], ground_hits[row],
                strict=True,
            ):
                active = rays.any(axis=1)
                count = int(active.sum())
                if not count:
                    continue
                ids = np.empty(count, dtype=np.int32)
                hits = np.empty(count)
                mujoco.mj_multiRay(  # pyright: ignore[reportAttributeAccessIssue]
                    m=model,
                    d=self.datas[env_id],
                    pnt=origin,
                    vec=rays[active].ravel(),
                    geomgroup=groups,
                    flg_static=include_ground,
                    bodyexclude=-1,
                    geomid=ids,
                    dist=hits,
                    normal=None,
                    nray=count,
                    cutoff=max_distance,
                )
                measured = (hits >= 0.) & (hits <= max_distance)
                result[active] = np.where(measured, hits, np.nan)
                if include_ground:
                    ground[active] = measured & np.isin(ids, self.floor_ids)

        return (
            torch.as_tensor(distances, device=origins.device, dtype=origins.dtype),
            torch.as_tensor(ground_hits, device=origins.device),
        )
