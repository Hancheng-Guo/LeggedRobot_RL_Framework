import torch
from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.utils.context import TaskContext, TaskStepResult
from envs.tasks.managers.action.base import ActionManager
from envs.tasks.managers.command.base import CommandManager
from envs.tasks.managers.observation.base import ObservationManager
from envs.tasks.managers.reward.base import RewardManager
from envs.tasks.managers.termination.base import TerminationManager
from utils.component import Component
from utils.param import update_attributes


class BaseTaskLogic(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:
        self.context = context

        self.num_envs: int
        self.model_context: ModelContext

        self.action_manager: ActionManager
        self.command_manager: CommandManager
        self.observation_manager: ObservationManager
        self.reward_manager: RewardManager
        self.termination_manager: TerminationManager


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
    ) -> None:

        self._build_action_manager(action_manager_config)
        self._build_command_manager(command_manager_config)
        self._build_observation_manager(observation_manager_config)
        self._build_reward_manager(reward_manager_config)
        self._build_termination_manager(termination_manager_config)


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


    def _build_observation_manager(
        self,
        observation_manager_config: dict,
    ) -> None:

        self.observation_manager = ObservationManager(
            num_envs=self.num_envs,
            context=self.context,
            model_context=self.model_context,
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


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        self.action_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.observation_manager.reset(env_ids)
        self.reward_manager.reset(env_ids)
        self.termination_manager.reset(env_ids)


    def build_task_context(
        self,
        state: dict[str, torch.Tensor],
        episode_step: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> TaskContext:
    
        if env_ids is None:
            command = self.command
            action = self.action
            last_action = self.last_action

        else:
            command = {
                name: value[env_ids]
                for name, value
                in self.command.items()
            }

            action = self.action[env_ids]
            last_action = self.last_action[env_ids]

        return TaskContext(
            state=state,
            command=command,
            action=action,
            last_action=last_action,
            episode_step=episode_step,
        )

    
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
