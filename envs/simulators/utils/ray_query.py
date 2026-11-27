"""Shared ground ray contracts and analytical plane intersections."""

import torch
from typing import Protocol


GROUND_RAY_QUERY = "ground_ray_query"


class RayQuery(Protocol):

    def __call__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return [env, group, point] distances for the supplied rays.

        Origins are [env, group, xyz], unit directions are [env, group, point, xyz],
        and misses are non-finite. Zero directions represent invalid poses.
        """
        ...


class HorizontalPlaneRayQuery:

    def __init__(
        self,
        height: float,
    ) -> None:
        self.height = height


    def __call__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
    ) -> torch.Tensor:

        vertical = directions[..., 2]
        crosses_plane = vertical.abs() > torch.finfo(origins.dtype).eps
        distance = (self.height - origins[..., 2, None]) / torch.where(
            crosses_plane, vertical, 1.,
        )
        return torch.where(
            crosses_plane & (distance >= 0.) & (distance <= max_distance),
            distance,
            torch.nan,
        )
