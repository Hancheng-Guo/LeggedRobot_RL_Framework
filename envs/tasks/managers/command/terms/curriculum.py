import torch

from envs.tasks.managers.command.terms.base import BaseCommandTerm
from envs.tasks.managers.command.terms.registry import register_command
from envs.tasks.managers.curriculum.types import CommandCurriculumSampler
from envs.tasks.utils.context import TaskContext


class CurriculumSampleOnReset(BaseCommandTerm):

    def __init__(
        self,
        curriculum_sampler: CommandCurriculumSampler,
        term_name: str,
        min_value: float,
        max_value: float,
        num_cell: int,
        group: str,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if min_value > max_value:
            raise ValueError("'min_value' cannot exceed 'max_value'.")
        if num_cell <= 0:
            raise ValueError("'num_cell' must be positive.")
        
        self.curriculum_sampler = curriculum_sampler
        self.curriculum_space = group
        self.dimension = term_name
        self.command_start = torch.zeros_like(self.command)
        self.command_end = torch.zeros_like(self.command)


    def update(
        self,
        task_context: TaskContext
    ) -> None:
        pass


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        command_start, command_end = self.curriculum_sampler.get_command(
            self.curriculum_space,
            self.dimension,
            env_ids,
        )
        
        if env_ids is None:
            self.command_start.copy_(command_start)
            self.command_end.copy_(command_end)
        else:
            self.command_start[env_ids] = command_start
            self.command_end[env_ids] = command_end

        self._update_command(env_ids)


    def _update_command(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self.command.copy_(
                self.command_start +
                (
                    (self.command_end - self.command_start)
                    * torch.rand_like(self.command_start)
                )
            )

        else:
            command_start = self.command_start[env_ids]
            command_end = self.command_end[env_ids]
            self.command[env_ids] = (
                command_start +
                (
                    (command_end - command_start)
                    * torch.rand_like(command_start)
                )
            )


# @register_command
# class LrpcSampleOnReset(CurriculumSampleOnReset):

#     curriculum_term_name = "lrpc_command_reward"


@register_command
class LpacSampleOnReset(CurriculumSampleOnReset):

    curriculum_term_name = "lpac_command_reward"


@register_command
class SymmetricLpacSampleOnReset(LpacSampleOnReset):

    def __init__(
        self,
        min_value: float,
        max_value: float,
        *args, **kwargs,
    ) -> None:

        if min_value < 0 or max_value < 0:
            raise ValueError("'min_value' and 'max_value' must be positive.")
        
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            *args, **kwargs
        )


    def _update_command(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        super()._update_command(env_ids)

        if env_ids is None:
            self.command *= torch.where(
                torch.rand_like(self.command) < 0.5,
                -torch.ones_like(self.command),
                torch.ones_like(self.command),
            )

        else:
            self.command[env_ids] *= torch.where(
                torch.rand_like(self.command[env_ids]) < 0.5,
                -torch.ones_like(self.command[env_ids]),
                torch.ones_like(self.command[env_ids]),
            )


@register_command
class FastLpacSampleOnReset(CurriculumSampleOnReset):

    curriculum_term_name = "fast_lpac_command_reward"


@register_command
class SymmetricFastLpacSampleOnReset(FastLpacSampleOnReset):

    def __init__(
        self,
        min_value: float,
        max_value: float,
        *args, **kwargs,
    ) -> None:

        if min_value < 0 or max_value < 0:
            raise ValueError("'min_value' and 'max_value' must be positive.")

        super().__init__(
            min_value=min_value,
            max_value=max_value,
            *args, **kwargs
        )


    def _update_command(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        super()._update_command(env_ids)

        if env_ids is None:
            self.command *= torch.where(
                torch.rand_like(self.command) < 0.5,
                -torch.ones_like(self.command),
                torch.ones_like(self.command),
            )

        else:
            self.command[env_ids] *= torch.where(
                torch.rand_like(self.command[env_ids]) < 0.5,
                -torch.ones_like(self.command[env_ids]),
                torch.ones_like(self.command[env_ids]),
            )
