import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.utils.context import RuntimeContext
from envs.tasks.managers.reward.base import RewardManager
from utils.component import Component


@dataclass
class TaskContext:
    state: dict[str, torch.Tensor]
    command: dict[str, torch.Tensor]
    action: torch.Tensor
    last_action: torch.Tensor
    episode_step: torch.Tensor


class BaseTaskLogic(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:
        self.context = context

        self.num_envs: int

        # self.action_manager: ActionManager
        # self.command_manager: CommandManager
        # self.observation_manager: ObservationManager
        self.reward_manager: RewardManager
        # self.termination_manager: TerminationManager


    def config_update(
        self,
        component: Component,
        num_envs: int,
        *args, **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self._build_managers(*args, **kwargs)


    def _build_managers(
        self,
        action_manager_config: dict,
        command_manager_config: dict,
        observation_manager_config: dict,
        reward_manager_config: dict,
        termination_manager_config: dict,
    ) -> None:

        # self.action_manager = ActionManager(
        #     num_envs=self.num_envs,
        #     context=self.context,
        #     **action_manager_config,
        # )

        # self.command_manager = CommandManager(
        #     num_envs=self.num_envs,
        #     context=self.context,
        #     **command_manager_config,
        # )

        # self.observation_manager = ObservationManager(
        #     num_envs=self.num_envs,
        #     context=self.context,
        #     **observation_manager_config,
        # )

        self.reward_manager = RewardManager(
            num_envs=self.num_envs,
            context=self.context,
            **reward_manager_config,
        )

        # self.termination_manager = TerminationManager(
        #     num_envs=self.num_envs,
        #     device=self.device,
        #     **termination_manager_config,
        # )


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        
        resolved_env_ids = self._resolve_env_ids(env_ids)

        self.action_manager.reset(resolved_env_ids)
        self.command_manager.reset(resolved_env_ids)
        self.observation_manager.reset(resolved_env_ids)
        self.reward_manager.reset(resolved_env_ids)
        self.termination_manager.reset(resolved_env_ids)


    def pre_step(
        self,
        action: torch.Tensor,
    ) -> None:
        self.action_manager.update(action)
        self.command_manager.update()


    def post_step(
        self,
        task_context: TaskContext,
    ) -> None:
        pass


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


    def process_action(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:

        return self.action_manager.process(action)
    

    def compute_observation(
        self,
        task_context: TaskContext,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        resolved_env_ids = self._resolve_env_ids(env_ids)

        return self.observation_manager.compute(
            task_context=task_context,
            env_ids=resolved_env_ids,
        )


    def compute_reward(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:

        return self.reward_manager.compute(
            task_context=task_context,
            command=self.command,
        )


    def check_terminated(
        self,
        task_context: TaskContext,
    ) -> torch.Tensor:

        return self.termination_manager.compute_terminated(
            task_context=task_context,
        )


    def _resolve_env_ids(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if env_ids is None:
            return torch.arange(
                self.num_envs,
                device=self.device,
            )

        return env_ids
        

    @property
    def command(self) -> torch.Tensor:
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
