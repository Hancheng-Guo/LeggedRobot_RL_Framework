"""Simulator capabilities consumed by scan terms."""

import torch
from typing import Protocol


class ScanRayProvider(Protocol):

    def resolve_scan_mounts(
        self,
        names: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mount origins and optical axes in base coordinates, in configured foot order."""
        ...


    def query_surface_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_distance: float,
        env_ids: torch.Tensor | None,
        *,
        include_robot: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nearest distances and whether each first hit belongs to the ground."""
        ...


def select_nearest_surface(
    ground: torch.Tensor,
    robot: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the first hit per ray, favoring the robot on ties."""

    ground_hit = ground.isfinite()
    if robot is None:
        return ground, ground_hit
    if robot.shape != ground.shape:
        raise ValueError("Unexpected robot ray distance shape.")
    blocked = robot.isfinite() & (~ground_hit | (robot <= ground))
    return torch.where(blocked, robot, ground), ground_hit & ~blocked
