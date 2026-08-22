import torch

from envs.tasks.managers.command.terms.base import BaseCommandTerm
from envs.tasks.managers.command.terms.registry import register_command
from envs.tasks.managers.curriculum.types import CommandCurriculumSampler
from envs.tasks.utils.context import TaskContext


@register_command
class CurriculumSampleOnReset(BaseCommandTerm):

    curriculum_term_name = "command_reward"

    def __init__(
        self,
        curriculum_sampler: CommandCurriculumSampler | None,
        term_name: str,
        min_value: float,
        max_value: float,
        num_bins: int,
        group: str,
        noise_scale: float,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if curriculum_sampler is None:
            raise ValueError(
                "CurriculumSampleOnReset requires a curriculum sampler."
            )
        if group is None:
            raise ValueError(
                "CurriculumSampleOnReset requires a curriculum group."
            )
        if min_value > max_value:
            raise ValueError("'min_value' cannot exceed 'max_value'.")
        if num_bins <= 0:
            raise ValueError("'num_bins' must be positive.")
        if noise_scale < 0.0:
            raise ValueError("'noise_scale' cannot be negative.")
        
        self.curriculum_sampler = curriculum_sampler
        self.curriculum_space = group
        self.dimension = term_name
        bin_width = (
            (max_value - min_value) / (num_bins - 1)
            if num_bins > 1
            else max_value - min_value
        )
        self.noise_std = noise_scale * bin_width
        self.command_center = torch.zeros_like(self.command)


    def update(
        self,
        task_context: TaskContext
    ) -> None:
        self._update_command_noise()


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        center = self.curriculum_sampler.get_command(
            self.curriculum_space,
            self.dimension,
            env_ids,
        )
        
        if env_ids is None:
            self.command_center.copy_(center)
        else:
            self.command_center[env_ids] = center

        self._update_command_noise(env_ids)


    def _update_command_noise(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self.command.copy_(
                self.command_center
                + torch.randn_like(self.command_center) * self.noise_std
            )
            return

        center = self.command_center[env_ids]
        self.command[env_ids] = (
            center + torch.randn_like(center) * self.noise_std
        )
