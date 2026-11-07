import torch
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from .managers import (
    TaskStateManager,
    ActionManager,
    CommandManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
    RandomizationManager,
)
from .utils import TaskContext, TaskStepResult
from envs.simulators.utils import ModelContext, SimulatorState
from .managers.randomization import RandomizationBackend
from utils import Component, RuntimeContext, update_attributes


class BaseTaskLogic(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:
        self.context = context
        self._simulator_requirements: dict[str, Any]

        self.num_envs: int
        self.model_context: ModelContext

        self.task_state_manager: TaskStateManager
        self.action_manager: ActionManager
        self.command_manager: CommandManager
        self.observation_manager: ObservationManager
        self.reward_manager: RewardManager
        self.termination_manager: TerminationManager
        self.randomization_manager: RandomizationManager


    def config_update(
        self,
        component: Component,
        num_envs: int,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:
        update_attributes(
            self,
            num_envs=num_envs,
            model_context=model_context,
        )
        self._build_managers(*args, **kwargs)


    def _build_managers(
        self,
        action_manager_config: dict,
        command_manager_config: dict,
        observation_manager_config: dict,
        reward_manager_config: dict,
        termination_manager_config: dict,
        constants: dict,
        task_state_manager_config: dict | None = None,
        randomization_manager_config: dict | None = None,
    ) -> None:

        self._build_randomization_manager(randomization_manager_config)
        self._build_task_state_manager(task_state_manager_config)
        self._build_action_manager(action_manager_config)
        self._build_command_manager(command_manager_config)
        self._build_observation_manager(observation_manager_config)
        self._build_reward_manager(reward_manager_config)
        self._build_termination_manager(termination_manager_config)


    def prepare(
        self,
        config: Mapping[str, Any],
    ) -> None:
        """Prepare requirements before a simulator/ModelContext exists.

        Subclasses may extend this hook for other pre-scene task resources.
        Only configuration is inspected here. All managers are built later in
        config_update once simulator metadata is available.
        """
        randomization_config = config.get("randomization_manager_config") or {}
        self._simulator_requirements = RandomizationManager.get_simulator_requirements(
            **randomization_config)


    def _build_randomization_manager(
        self,
        randomization_manager_config: Mapping[str, Any] | None,
    ) -> None:
        """Build randomization with the other managers; retain unchanged state."""

        selected = randomization_manager_config or {}
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


    def get_simulator_requirements(self) -> dict[str, Any]:
        """Expose physical configuration requirements without manager internals.

        Keys must be distinct from simulator YAML fields; duplicates are errors.
        """
        return dict(self._simulator_requirements)


    def bind(
        self,
        simulator: RandomizationBackend,
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


    def abort_simulator_reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restore task-owned physical changes when a reset fails."""
        self.randomization_manager.abort_simulator_reset(env_ids)


    def _build_action_manager(
        self,
        action_manager_config: dict,
    ) -> None:

        self.action_manager = ActionManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            **action_manager_config,
        )


    def _build_command_manager(
        self,
        command_manager_config: dict,
    ) -> None:

        self.command_manager = CommandManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            **command_manager_config,
        )


    def _build_task_state_manager(
        self,
        task_state_manager_config: dict | None,
    ) -> None:

        self.task_state_manager = TaskStateManager(
            num_envs=self.num_envs,
            context=self.context,
            **(task_state_manager_config or {}),
        )


    def _build_observation_manager(
        self,
        observation_manager_config: dict,
    ) -> None:

        self.observation_manager = ObservationManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            command_dim=self.command_manager.output_dim,
            action_dim=self.action_manager.input_dim,
            **observation_manager_config,
        )


    def _build_reward_manager(
        self,
        reward_manager_config: dict,
    ) -> None:

        self.reward_manager = RewardManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            **reward_manager_config,
        )


    def _build_termination_manager(
        self,
        termination_manager_config: dict,
    ) -> None:

        self.termination_manager = TerminationManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
            **termination_manager_config,
        )


    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Save persistent training statistics, excluding unfinished episodes."""
        return {}


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Task checkpoint must be a mapping.")
        if state:
            raise ValueError("This task does not support the saved checkpoint state.")


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        self.task_state_manager.reset(env_ids)
        self.action_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.observation_manager.reset(env_ids)
        self.reward_manager.reset(env_ids)
        self.termination_manager.reset(env_ids)


    def build_task_context(
        self,
        state: SimulatorState,
        episode_step: torch.Tensor,
        step_dt: float,
        env_ids: torch.Tensor | None = None,
    ) -> TaskContext:
    
        if env_ids is None:
            command = self.command
            last_command = self.last_command
            action = self.action
            last_action = self.last_action

        else:
            command = {
                name: value[env_ids]
                for name, value
                in self.command.items()
            }
            last_command = {
                name: value[env_ids]
                for name, value
                in self.last_command.items()
            }

            action = self.action[env_ids]
            last_action = self.last_action[env_ids]

        return TaskContext(
            state=state,
            command=command,
            last_command=last_command,
            action=action,
            last_action=last_action,
            episode_step=episode_step,
            step_dt=step_dt,
            env_ids=env_ids,
            task_state=self.task_state_manager.values(env_ids),
        )


    def update_task_state(
        self,
        task_context: TaskContext
    ) -> None:
        self.task_state_manager.update(task_context)

    
    def pre_step(
        self,
    ) -> dict:

        return {}


    def post_step(
        self,
        task_context: TaskContext,
        step_result: TaskStepResult,
    ) -> dict:

        command_update_info = self.command_manager.update(task_context)

        return command_update_info


    def collect_metrics(self) -> dict:
        """Collect task metrics after completed episodes have been settled."""
        return {}


    def process_action(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        return self.action_manager.process(action)
    

    def compute_observation(
        self,
        task_context: TaskContext,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:

        return self.observation_manager.compute(
            task_context=task_context,
            env_ids=env_ids,
        )


    def compute_reward(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:

        return self.reward_manager.compute(
            task_context=task_context,
        )


    def check_terminated(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:

        return self.termination_manager.compute(
            task_context=task_context,
        )


    @property
    def command(self) -> dict[str, torch.Tensor]:
        return self.command_manager.command


    @property
    def last_command(self) -> dict[str, torch.Tensor]:
        return self.command_manager.last_command


    @property
    def action(self) -> torch.Tensor:
        return self.action_manager.action


    @property
    def last_action(self) -> torch.Tensor:
        return self.action_manager.last_action
        

    @property
    def control(self) -> torch.Tensor:
        return self.action_manager.control


    @property
    def last_control(self) -> torch.Tensor:
        return self.action_manager.last_control


    @property
    def observation_dim(self) -> int:
        return self.observation_manager.output_dim


    @property
    def observation_slices(self) -> dict[str, slice]:
        return dict(self.observation_manager.term_slices)
