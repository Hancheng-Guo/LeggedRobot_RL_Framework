import torch
import numpy as np
from typing import Any
from abc import ABC, abstractmethod

from app.utils.context import RuntimeContext


class BaseEnv(ABC):

    SUPPORTS_CONCURRENT_INSTANCES = True

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:
        self.num_envs: int
        self.context = context


    @property
    def supports_concurrent_instances(self) -> bool:
        """Whether another environment may coexist in this process."""
        return self.SUPPORTS_CONCURRENT_INSTANCES


    @property
    def render_mode(self) -> str | None:
        """Rendering mode used by this environment, if any."""
        return None


    def check_render_mode(self) -> bool:
        """Whether playback has a configured render mode."""
        return self.render_mode is not None


    @property
    def playback_env_index(self) -> int:
        """Environment whose episode bounds a playback recording."""
        return 0


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
        """Advance the vectorized environment by one control step.

        Tensor values in ``info`` may be views of reusable internal buffers.
        They are valid for synchronous consumption during the current step,
        but callers that retain them across another call to ``step`` must
        clone them first.

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
    def render_fps(self) -> float:
        """Frequency of frames returned by consecutive environment steps."""
        raise NotImplementedError

    
    @property
    @abstractmethod
    def observation_dim(self) -> int:
        raise NotImplementedError


    @property
    @abstractmethod
    def observation_slices(self) -> dict[str, slice]:
        """Named slices in the flattened observation tensor."""
        raise NotImplementedError


    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError
