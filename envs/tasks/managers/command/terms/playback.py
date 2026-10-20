"""Temporary command term used during interactive playback."""

import torch
from collections.abc import Mapping
from typing import Protocol

from .base import BaseCommandTerm
from envs.tasks.utils import TaskContext


class PlaybackCommandSource(Protocol):
    def values(self) -> Mapping[str, float]: ...


class PlaybackCommandTerm(BaseCommandTerm):
    """Keep the original term active, replacing one environment's value."""

    def __init__(
        self,
        original: BaseCommandTerm,
        source: PlaybackCommandSource,
        command_name: str,
        env_index: int,
        fixed_value: tuple[float, ...] | None = None,
    ) -> None:

        self.original = original
        self.source = source
        self.command_name = command_name
        self.env_index = env_index
        self.command = original.command
        self.dim = original.dim
        self.num_envs = original.num_envs
        self.context = original.context
        self.model_context = original.model_context
        self.fixed_value = (
            torch.tensor(fixed_value, dtype=self.command.dtype, device=self.command.device)
            if fixed_value is not None else None
        )


    def _apply(self) -> None:
        if self.fixed_value is None:
            self.command[self.env_index, 0] = self.source.values()[self.command_name]
        else:
            self.command[self.env_index] = self.fixed_value


    def update(
        self,
        task_context: TaskContext
    ) -> None:

        self.original.update(task_context)
        self._apply()


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        self.original.reset(env_ids)
        if env_ids is None or bool((env_ids == self.env_index).any().item()):
            self._apply()
