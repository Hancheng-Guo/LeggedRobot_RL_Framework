"""Temporary command term used during interactive playback."""

import torch
from collections.abc import Mapping
from typing import Protocol

from .base import BaseCommandTerm
from envs.tasks.utils import TaskContext


class PlaybackCommandSource(Protocol):
    def values(self) -> Mapping[str, float]: ...


class PlaybackCommandTerm(BaseCommandTerm):

    RESET_STATE_FIELDS = ('env_index',)
    """Keep the original term active, replacing one environment's value."""

    def __init__(
        self,
        name: str,
        original: BaseCommandTerm,
        source: PlaybackCommandSource,
        env_index: int,
        fixed_value: tuple[float, ...] | None = None,
    ) -> None:

        super().__init__(
            name=name, num_envs=original.num_envs, context=original.context,
            model_context=original.model_context, dim=original.dim,
        )
        self.original = original
        self.source = source
        self.env_index = env_index
        self.command = original.command
        self.fixed_value = (
            torch.tensor(fixed_value, dtype=self.command.dtype, device=self.command.device)
            if fixed_value is not None else None
        )


    def _apply(self) -> None:
        if self.fixed_value is None:
            self.command[self.env_index, 0] = self.source.values()[self.name]
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
