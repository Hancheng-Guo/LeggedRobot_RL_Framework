import torch
from typing import Any
from collections.abc import Mapping

from .base import BaseTaskLogic
from .managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
    TaskStateManager,
    CommandManager,
    CurriculumManager,
    RandomizationManager,
)
from .managers.randomization import RandomizationBackend
from .managers.curriculum.types import CurriculumSpaceConfigs, CurriculumSpaceProvider
from .utils import TaskContext, TaskStepResult


class LocomotionTaskLogic(BaseTaskLogic):

    CURRICULUM_SPACE_PROVIDERS: dict[str, type[CurriculumSpaceProvider]] = {
        "action_manager_config": ActionManager,
        "command_manager_config": CommandManager,
        "observation_manager_config": ObservationManager,
        "reward_manager_config": RewardManager,
        "termination_manager_config": TerminationManager,
        "task_state_manager_config": TaskStateManager,
        "randomization_manager_config": RandomizationManager,
    }

    @classmethod
    def get_curriculum_space_configs(
        cls,
        manager_configs: Mapping[str, Any],
    ) -> list[CurriculumSpaceConfigs]:
        """Collect descriptions before managers with curriculum dependencies exist."""
        return [manager_type.get_curriculum_space_config(manager_configs.get(name))
                for name, manager_type in cls.CURRICULUM_SPACE_PROVIDERS.items()]


    def __init__(self, context):

        super().__init__(context)

        self.curriculum_manager: CurriculumManager
        self.randomization_manager: RandomizationManager
        self._simulator_requirements: dict[str, Any]
        # self.event_manager: EventManager


    def _build_managers(
        self,
        action_manager_config: dict,
        command_manager_config: dict,
        observation_manager_config: dict,
        reward_manager_config: dict,
        termination_manager_config: dict,
        task_state_manager_config: dict,
        curriculum_manager_config: dict | None = None,
        event_manager_config: dict | None = None,
        randomization_manager_config: dict | None = None,
        *args, **kwargs,
    ) -> None:

        if event_manager_config is not None and event_manager_config != {}:
            raise NotImplementedError(
                "Event manager is not implemented; omit its config or use {}."
            )

        manager_configs = {
            "action_manager_config": action_manager_config,
            "command_manager_config": command_manager_config,
            "observation_manager_config": observation_manager_config,
            "reward_manager_config": reward_manager_config,
            "termination_manager_config": termination_manager_config,
            "randomization_manager_config": randomization_manager_config,
            "task_state_manager_config": task_state_manager_config,
        }
        self._build_curriculum_manager(curriculum_manager_config, manager_configs)
        self._build_randomization_manager(randomization_manager_config)
        # self._build_event_manager(event_manager_config)
        
        super()._build_managers(
            action_manager_config=action_manager_config,
            command_manager_config=command_manager_config,
            observation_manager_config=observation_manager_config,
            reward_manager_config=reward_manager_config,
            termination_manager_config=termination_manager_config,
            task_state_manager_config=task_state_manager_config,
            *args, **kwargs,
        )


    def prepare(
        self,
        *,
        config: Mapping[str, Any],
        **kwargs,
    ) -> None:
        """Prepare requirements before a simulator/ModelContext exists.

        Subclasses may extend this hook for other pre-scene task resources.
        Only configuration is inspected here. All managers are built later in
        config_update once simulator metadata is available.
        """
        randomization_config = config.get("randomization_manager_config")
        if randomization_config is None:
            randomization_config = {}
        self._simulator_requirements = (
            RandomizationManager.get_simulator_requirements(
                **randomization_config
            )
        )


    def _build_randomization_manager(
        self,
        randomization_manager_config: Mapping[str, Any] | None,
    ) -> None:
        """Build randomization with the other managers; retain unchanged state."""

        selected = (
            {}
            if randomization_manager_config is None
            else randomization_manager_config
        )
        if (
            hasattr(self, "randomization_manager")
            and self.randomization_manager.matches_config(self.num_envs, **selected)
        ):
            return

        self.randomization_manager = RandomizationManager(
            num_envs=self.num_envs,
            context=self.context,
            **selected,
        )


    def _build_curriculum_manager(
        self,
        curriculum_manager_config: dict | None,
        manager_configs: dict[str, Any],
        *args, **kwargs,
    ) -> None:

        selected = (
            {}
            if curriculum_manager_config is None
            else curriculum_manager_config
        )
        self.curriculum_manager = CurriculumManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            manager_configs=manager_configs,
            space_configs=self.get_curriculum_space_configs(manager_configs),
            *args, **kwargs,
            **selected,
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


    def get_simulator_requirements(self) -> dict[str, Any]:
        """Expose physical configuration requirements without manager internals.

        Keys must be distinct from simulator YAML fields; duplicates are errors.
        """
        return dict(self._simulator_requirements)


    def bind(
        self,
        *,
        simulator: RandomizationBackend,
        **kwargs,
    ) -> None:
        """Bind task resources after simulator and task configuration complete."""
        self.randomization_manager.bind(simulator)


    def before_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Prepare task changes before the environment resets physics."""
        self.randomization_manager.before_simulator_reset(env_ids)


    def after_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Apply task changes after the environment resets physics."""
        self.randomization_manager.after_simulator_reset(env_ids)


    def abort_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Restore task-owned physical changes when a reset fails."""
        self.randomization_manager.abort_simulator_reset(env_ids)


    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "curriculum": self.curriculum_manager.checkpoint_state_dict()
        }


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Task checkpoint must be a mapping.")
        if set(state) != {"curriculum"}:
            raise ValueError("Locomotion task checkpoint must contain curriculum state.")
        self.curriculum_manager.load_checkpoint_state_dict(state["curriculum"])


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Reset curriculum episode bookkeeping and task managers only."""

        self.curriculum_manager.reset(env_ids)
        self.curriculum_manager.resample(env_ids)
        # self.event_manager.reset(env_ids)
        # Environment reset hooks have already applied physical randomization.
        super().reset(env_ids)


    def set_training_mode(self, training: bool) -> None:
        self.curriculum_manager.set_training_mode(training)



    def post_step(
        self,
        task_context: TaskContext,
        step_result: TaskStepResult,
    ) -> dict:

        curriculum_info = self.curriculum_manager.update(
            raw_term_rewards=self.reward_manager.raw_term_rewards,
            term_rewards=self.reward_manager.term_rewards,
            total_reward=self.reward_manager.total_reward,
        )
        manager_info = super().post_step(task_context, step_result)

        return curriculum_info | manager_info


    def collect_metrics(self) -> dict:
        return (
            super().collect_metrics()
            | self.curriculum_manager.collect_metrics()
        )
