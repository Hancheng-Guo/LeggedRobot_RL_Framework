import torch
from typing import Any
from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext


class BaseEnv(ABC):

    def __init__(
        self,
        context: RuntimeContext,
    ):
        self.num_envs: int
        self.context = context


    @abstractmethod
    def config_update(self, *args, **kwargs):
        pass


    @abstractmethod
    def reset(self):
        pass


    @abstractmethod
    def step(
        self,
        actions: torch.Tensor,
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
        pass

    @abstractmethod
    def close(self) -> None:
        pass