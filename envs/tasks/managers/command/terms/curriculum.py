import torch
from collections.abc import Mapping
from typing import Any

from .base import BaseCommandTerm
from .registry import register_command
from envs.tasks.managers.curriculum import (
    CurriculumTermProvider, resolve_curriculum_samplers,
    CurriculumSpaceConfigs, CurriculumSpaceConfig, ContinuousDimensionConfig,
)
from envs.tasks.utils import TaskContext


@register_command
class Curriculum(BaseCommandTerm):

    RESET_STATE_FIELDS = ('command_start', 'command_end')

    @classmethod
    def get_curriculum_references(
        cls,
        params: Mapping[str, Any],
    ) -> set[str]:

        name = params.get("curriculum")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Curriculum command sampling requires a non-empty 'curriculum' reference."
            )
        return {name}


    @classmethod
    def get_curriculum_space_config(
        cls,
        name: str,
        params: Mapping[str, Any],
    ) -> CurriculumSpaceConfigs:

        curriculum_name, = cls.get_curriculum_references(params)
        if "min_value" not in params or "max_value" not in params:
            raise ValueError(f"Command term '{name}' requires min_value and max_value.")
        return {
            curriculum_name: CurriculumSpaceConfig(
                dimensions={
                    name: ContinuousDimensionConfig(
                        min_value=params["min_value"],
                        max_value=params["max_value"],
                        num_cell=params.get("num_cell", 1),
                        dim=params.get("dim", 1),
                    ),
                }
            )
        }


    def __init__(
        self,
        curriculum: str,
        min_value: float,
        max_value: float,
        num_cell: int,
        curriculum_manager: CurriculumTermProvider,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if min_value > max_value:
            raise ValueError("'min_value' cannot exceed 'max_value'.")
        if num_cell <= 0:
            raise ValueError("'num_cell' must be positive.")
        
        curriculum_names = self.get_curriculum_references({"curriculum": curriculum, **kwargs})
        self.curriculum_sampler = resolve_curriculum_samplers(
            curriculum_names, curriculum_manager, f"Command term {self.name!r}",
        )[curriculum]
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
        
        command_start, command_end = self.curriculum_sampler.get_bounds(
            self.name,
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


@register_command
class SymmetricCurriculum(Curriculum):

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
