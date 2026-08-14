import torch
from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseTaskLogic(ABC):

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:

        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        # Managers
        self.command_manager = None
        self.observation_manager = None
        self.reward_manager = None
        self.termination_manager = None
        self.curriculum_manager = None

        # Task states
        self.episode_length = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self._setup()

    def _setup(self) -> None:
        self._build_managers()

    @abstractmethod
    def _build_managers(self) -> None:
        """Create task-specific managers."""
        raise NotImplementedError

    def reset(
        self,
        env_ids: Iterable[int] | torch.Tensor | None = None,
    ) -> None:

        env_ids = self._resolve_env_ids(
            env_ids
        )

        self.episode_length[env_ids] = 0

        if self.command_manager is not None:
            self.command_manager.reset(env_ids)

        if self.observation_manager is not None:
            self.observation_manager.reset(env_ids)

        if self.reward_manager is not None:
            self.reward_manager.reset(env_ids)

        if self.termination_manager is not None:
            self.termination_manager.reset(env_ids)

        if self.curriculum_manager is not None:
            self.curriculum_manager.reset(env_ids)

    def process_action(
        self,
        action: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        """
        Convert policy action into simulator control.

        Example:
            normalized action [-1, 1]
                ->
            target joint position / torque / velocity
        """

        return action

    def compute_observation(
        self,
        state: Any,
    ) -> torch.Tensor:

        if self.observation_manager is None:
            raise RuntimeError(
                "observation_manager is not initialized."
            )

        return self.observation_manager.compute(
            state=state,
            command=self.command,
        )

    def compute_reward(
        self,
        state: Any,
    ) -> torch.Tensor:

        if self.reward_manager is None:
            raise RuntimeError(
                "reward_manager is not initialized."
            )

        return self.reward_manager.compute(
            state=state,
            command=self.command,
        )

    def compute_terminated(
        self,
        state: Any,
    ) -> torch.Tensor:

        if self.termination_manager is None:
            return torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

        return self.termination_manager.compute_terminated(
            state=state,
        )

    def compute_truncated(
        self,
        state: Any,
    ) -> torch.Tensor:

        if self.termination_manager is None:
            return torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

        return self.termination_manager.compute_truncated(
            state=state,
            episode_length=self.episode_length,
        )

    def step(
        self,
    ) -> None:
        """
        Update task-side states once per environment step.
        """

        self.episode_length += 1

        if self.command_manager is not None:
            self.command_manager.update()

        if self.curriculum_manager is not None:
            self.curriculum_manager.update()

    @property
    def command(self) -> torch.Tensor | None:

        if self.command_manager is None:
            return None

        return self.command_manager.command

    def get_info(self) -> dict[str, Any]:

        info = {
            "episode_length": self.episode_length,
        }

        if self.command_manager is not None:
            info["command"] = self.command

        return info

    def _resolve_env_ids(
        self,
        env_ids: Iterable[int] | torch.Tensor | None,
    ) -> torch.Tensor:

        if env_ids is None:
            return torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )

        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(
                device=self.device,
                dtype=torch.long,
            )

        return torch.tensor(
            list(env_ids),
            device=self.device,
            dtype=torch.long,
        )