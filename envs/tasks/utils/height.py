"""Shared height configuration and signal utilities for task terms."""
import torch

from .context import TaskContext


def validate_height_reference(value: str) -> str:
    if value not in ("ground", "world"):
        raise ValueError(
            "height_reference must be 'ground' or 'world'."
        )
    return value


def base_height(
    task_context: TaskContext,
    qpos_id: int,
    height_reference: str = "ground",
) -> torch.Tensor:
    height = task_context.state.qpos[:, qpos_id]
    if height_reference == "ground" and task_context.state.base_ground_height is not None:
        height = height - task_context.state.base_ground_height
        if task_context.state.ground_valid is not None:
            height = torch.where(task_context.state.ground_valid, height, 0.)
    return height


def foot_heights(
    task_context: TaskContext,
    foot_geom_ids: torch.Tensor,
    height_reference: str = "ground",
) -> torch.Tensor:
    height = task_context.state.geom_xpos[:, foot_geom_ids, 2]
    if height_reference == "ground" and task_context.state.foot_ground_heights is not None:
        height = height - task_context.state.foot_ground_heights
        if task_context.state.ground_valid is not None:
            height = torch.where(task_context.state.ground_valid[:, None], height, 0.)
    return height


def mask_ground_signal(
    task_context: TaskContext,
    value: torch.Tensor,
    height_reference: str,
) -> torch.Tensor:
    """Rows without valid ground have no defined ground-relative reward/height test."""
    if height_reference == "ground" and task_context.state.ground_valid is not None:
        return torch.where(task_context.state.ground_valid, value, torch.zeros_like(value))
    return value
