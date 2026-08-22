import torch

from envs.tasks.managers.command.terms.base import BaseCommandTerm
from envs.tasks.managers.command.terms.registry import register_command
from envs.tasks.utils.context import TaskContext


@register_command
class UniformOnReset(BaseCommandTerm):

    def __init__(
        self,
        min_value: float,
        max_value: float,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if min_value > max_value:
            raise ValueError(
                "'min_value' cannot be greater than 'max_value'."
            )

        self.min_value = min_value
        self.max_value = max_value


    def update(
        self,
        task_context: TaskContext,
    ) -> None:
        pass


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:

            self.command.uniform_(
                self.min_value,
                self.max_value,
            )

            return

        self.command[env_ids] = (
            torch.rand(
                (env_ids.numel(), 1),
                dtype=self.context.dtype,
                device=self.context.device,
            )
            * (self.max_value - self.min_value)
            + self.min_value
        )
