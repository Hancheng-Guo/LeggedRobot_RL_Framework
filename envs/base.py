import torch
import numpy as np
from typing import Any
from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext


class BaseEnv(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:
        self.num_envs: int
        self.context = context


    @abstractmethod
    def config_update(
        self,
        num_envs: int = 1,
        *args, **kwargs
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def reset(self) -> torch.Tensor:
        raise NotImplementedError


    @abstractmethod
    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any]
    ]:
        """
        Returns:
            next_obs,
            transition_next_obs,
            reward,
            terminated,
            truncated,
            info,
        """
        raise NotImplementedError


    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


    @abstractmethod
    def render(self) -> np.ndarray | None:
        raise NotImplementedError

    
    @property
    @abstractmethod
    def observation_dim(self) -> int:
        raise NotImplementedError


    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError
