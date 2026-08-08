import torch
from abc import ABC, abstractmethod


class BaseEnv(ABC):

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.device = device


    @abstractmethod
    def reset(self):
        """
        Returns:
            observation
        """
        pass


    @abstractmethod
    def step(
        self,
        actions: torch.Tensor,
    ):
        """
        Args:
            actions:
                [num_envs, action_dim]

        Returns:
            observation,
            reward,
            terminated,
            info
        """
        pass