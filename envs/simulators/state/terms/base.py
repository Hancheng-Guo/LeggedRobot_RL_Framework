from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Mapping

from ..state import SimulatorState
from utils import CheckpointStateOwner


class BaseSimulatorStateTerm(CheckpointStateOwner, ABC):

    required_capabilities: frozenset[str] = frozenset()
    output_shapes: dict[str, tuple[int, ...]]


    def __init__(
        self,
        name: str,
        **kwargs,
    ) -> None:
        self.name = name


    @abstractmethod
    def compute(
        self,
        state: SimulatorState,
        env_ids: torch.Tensor | None,
    ) -> Mapping[str, torch.Tensor | None]:
        """Return values for fields declared in output_shapes."""
        raise NotImplementedError
