import torch
from typing import Any

from app.utils.context import RuntimeContext
from envs.tasks.base import BaseTaskLogic
from envs.tasks.managers.command.base import CommandManager
from envs.tasks.managers.curriculum.base import CurriculumManager
from envs.tasks.utils.context import TaskContext, TaskStepResult


class LocomotionTaskLogic(BaseTaskLogic):

    def __init__(self, context):

        super().__init__(context)

        self.curriculum_manager: CurriculumManager
        # self.event_manager: EventManager
        # self.randomization_manager: RandomizationManager


    def _build_managers(
        self,
        action_manager_config: dict,
        command_manager_config: dict,
        observation_manager_config: dict,
        reward_manager_config: dict,
        termination_manager_config: dict,
        curriculum_manager_config: dict,
        event_manager_config: dict,
        randomization_manager_config: dict,
        *args, **kwargs,
    ) -> None:

        manager_configs = {
            "action_manager_config": action_manager_config,
            "command_manager_config": command_manager_config,
            "observation_manager_config": observation_manager_config,
            "reward_manager_config": reward_manager_config,
            "termination_manager_config": termination_manager_config,
        }
        self._build_curriculum_manager(
            curriculum_manager_config=curriculum_manager_config,
            manager_configs=manager_configs,
        )
        # self._build_event_manager(event_manager_config)
        # self._build_randomization_manager(randomization_manager_config)
        
        super()._build_managers(
            action_manager_config=action_manager_config,
            command_manager_config=command_manager_config,
            observation_manager_config=observation_manager_config,
            reward_manager_config=reward_manager_config,
            termination_manager_config=termination_manager_config,
            *args, **kwargs,
        )


    def _build_curriculum_manager(
        self,
        curriculum_manager_config: dict,
        manager_configs: dict[str, dict[str, Any]],
        *args, **kwargs,
    ) -> None:

        self.curriculum_manager = CurriculumManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            manager_configs=manager_configs,
            *args, **kwargs,
            **curriculum_manager_config,
        )


    def _build_command_manager(
        self,
        command_manager_config: dict,
    ) -> None:

        self.command_manager = CommandManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            curriculum_manager=self.curriculum_manager,
            **command_manager_config,
        )


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        self.curriculum_manager.reset(env_ids)
        # self.event_manager.reset(env_ids)
        # self.randomization_manager.reset(env_ids)
        super().reset(env_ids)


    def post_step(
        self,
        task_context: TaskContext,
        step_result: TaskStepResult,
    ) -> dict:

        curriculum_info = self.curriculum_manager.update(
            step_result.reward
        )
        manager_info = super().post_step(task_context, step_result)

        return curriculum_info | manager_info
