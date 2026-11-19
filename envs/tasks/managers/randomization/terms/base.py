"""Lifecycle contract shared by task randomization terms."""
import torch
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

from envs.tasks.managers.curriculum import CurriculumSpaceConfigs
from envs.tasks.utils import StateOwner
from utils import RuntimeContext


class RandomizationBackend(Protocol):
    context: RuntimeContext
    num_envs: int


class BaseRandomizationTerm(StateOwner, ABC):

    def __init__(
        self,
        name: str,
        num_envs: int,
        context: RuntimeContext,
        *args, **kwargs,
    ):

        super().__init__(name=name)

        self.num_envs = num_envs
        self.context = context


    @classmethod
    def get_simulator_requirements(
        cls,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {}


    @abstractmethod
    def matches_config(
        self,
        params: Mapping[str, Any],
    ) -> bool:
        """Compare incoming configuration with this term's configuration."""
        raise NotImplementedError


    @classmethod
    def get_curriculum_space_config(
        cls,
        name: str,
        params: Mapping[str, Any],
    ) -> CurriculumSpaceConfigs:
        return {}


    @classmethod
    def get_curriculum_references(
        cls,
        params: Mapping[str, Any],
    ) -> set[str]:
        return set()


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
