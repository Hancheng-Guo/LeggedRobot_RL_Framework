"""Lifecycle contract shared by task randomization terms."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

import torch
from utils import RuntimeContext


class RandomizationBackend(Protocol):
    context: RuntimeContext
    num_envs: int


class BaseRandomizationTerm(ABC):

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        **params
    ):
        self.num_envs = num_envs
        self.context = context
        self.config = self.normalize_config(params)


    @classmethod
    @abstractmethod
    def normalize_config(
        cls,
        params: Mapping[str, Any],
    ) -> Any:
        raise NotImplementedError


    @classmethod
    def get_simulator_requirements(
        cls,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {}


    def matches_config(
        self,
        params: Mapping[str, Any],
    ) -> bool:
        return self.config == self.normalize_config(params)


    def bind(
        self,
        backend: RandomizationBackend,
    ) -> None:
        pass


    def before_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass


    def after_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        pass


    def commit_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Commit validated logical state after all terms apply; must not fail.

        Perform fallible validation and physical work in the before/after hooks.
        """
        pass


    def abort_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Undo a pending reset; must be safe to call more than once."""
        pass
