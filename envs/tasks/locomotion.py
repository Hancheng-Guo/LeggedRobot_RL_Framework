import torch

from app.utils.context import RuntimeContext
from envs.tasks.base import BaseTaskLogic


class LocomotionTaskLogic(BaseTaskLogic):

    def __init__(self, context):
        super().__init__(context)

        # self.curriculum_manager: CurriculumManager
        # self.event_manager: EventManager
        # self.randomization_manager: RandomizationManager

    def _build_managers(
        self,
        curriculum_manager_config: dict,
        event_manager_config: dict,
        randomization_manager_config: dict,
        *args, **kwargs,
    ) -> None:
        
        super()._build_managers(*args, **kwargs)

        # self.curriculum_manager = CurriculumManager(
        #     num_envs=self.num_envs,
        #     device=self.device,
        #     **curriculum_manager_config,
        # )

        # self.event_manager = EventManager(
        #     num_envs=self.num_envs,
        #     device=self.device,
        #     **event_manager_config,
        # )

        # self.randomization_manager = RandomizationManager(
        #     num_envs=self.num_envs,
        #     device=self.device,
        #     **randomization_manager_config,
        # )


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        super().reset(env_ids)
        
        resolved_env_ids = self._resolve_env_ids(env_ids)

        # self.curriculum_manager.reset(resolved_env_ids)
        # self.event_manager.reset(resolved_env_ids)
        # self.randomization_manager.reset(resolved_env_ids)
