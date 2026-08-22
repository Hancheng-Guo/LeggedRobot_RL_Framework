import torch
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from envs.simulators.utils.context import ModelContext


class BaseSimulator(ABC):

    def __init__(self) -> None:

        self.num_envs: int
        self.model_path: Path
        self.sim_dt: float
        self.frame_skip: int
        self.render_mode: str | None
        self.model_context: ModelContext


    @abstractmethod
    def config_update(
        self,
        num_envs: int | None = None,
        model_path: Path | str | None = None,
        sim_dt: float | None = None,
        frame_skip: int | None = None,
        render_mode: str | None = None,
        *args, **kwargs
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def _build_model_context(self) -> None:
        raise NotImplementedError


    @property
    def control_dt(self) -> float:

        return (
            self.sim_dt *
            self.frame_skip
        )


    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def step(
        self,
        action: torch.Tensor,
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def render(
        self,
    ) -> np.ndarray | None:
        raise NotImplementedError


    @abstractmethod
    def close(
        self,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError
