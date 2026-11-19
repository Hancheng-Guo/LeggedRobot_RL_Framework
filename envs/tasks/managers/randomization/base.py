"""Task randomization manager: registry-based construction and lifecycle dispatch."""
import torch
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from .terms.base import RandomizationBackend
from .terms import BaseRandomizationTerm, get_randomization_class
from envs.tasks.managers.curriculum import (
    CurriculumSpaceConfig, CurriculumSpaceConfigs,
)
from envs.tasks.utils import StateManager
from utils import RuntimeContext


class RandomizationManager(StateManager):

    CHECKPOINT_KEEP_EMPTY_TERMS = True

    @classmethod
    def get_curriculum_space_config(
        cls,
        config: Mapping[str, Any] | None,
    ) -> CurriculumSpaceConfigs:

        if config is None:
            return {}

        result: CurriculumSpaceConfigs = {}
        for name, params in cls._term_configs(config.get("terms")).items():
            term_cls = get_randomization_class(name)
            descriptions = term_cls.get_curriculum_space_config(name, params)
            for reference, space in descriptions.items():
                previous = result.get(reference, CurriculumSpaceConfig({}))
                if previous.dimensions.keys() & space.dimensions.keys():
                    raise ValueError(
                        "Randomization terms contribute duplicate curriculum dimensions."
                    )
                result[reference] = CurriculumSpaceConfig(
                    dimensions=previous.dimensions | space.dimensions,
                    filters=previous.filters + space.filters,
                    source_config={"randomization": deepcopy(dict(config))},
                    persistent_assignments=(
                        previous.persistent_assignments or space.persistent_assignments
                    ),
                )
        return result


    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        terms: Mapping[str, Any] | None = None,
        *args, **kwargs,
    ) -> None:

        self.num_envs = num_envs
        self.context = context
        self.terms: dict[str, BaseRandomizationTerm] = {}
        self._build_terms(terms, *args, **kwargs)
        self._bound = False


    def config_update(
        self,
        **kwargs,
    ) -> None:
        """Refresh dependencies on a reused manager, retaining allocations and backend binding."""

        for term in self.terms.values():
            term.config_update(**kwargs)


    def set_training_mode(self, training: bool) -> None:
        for term in self.terms.values():
            term.set_training_mode(training)


    def _build_terms(
        self,
        terms: Mapping[str, Any] | None,
        *args, **kwargs,
    ) -> None:

        for name, params in self._term_configs(terms).items():
            if name in self.terms:
                raise ValueError(f"Randomization term '{name}' already exists.")

            cls = get_randomization_class(name)
            self.terms[name] = cls(
                name=name,
                num_envs=self.num_envs,
                context=self.context,
                **kwargs,
                **params,
            )


    @staticmethod
    def _term_configs(
        terms: Mapping[str, Any] | None
    ) -> dict[str, dict[str, Any]]:

        if terms is None:
            return {}
        if not isinstance(terms, Mapping):
            raise TypeError("Randomization 'terms' must be a mapping.")

        result = {}
        for name, params in terms.items():
            if params is not None and not isinstance(params, Mapping):
                raise TypeError(f"Config of randomization term '{name}' must be a mapping.")
            result[name] = dict(params or {})
        return result


    @classmethod
    def get_simulator_requirements(
        cls,
        terms: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:

        requirements = {}
        for name, params in cls._term_configs(terms).items():
            for key, value in get_randomization_class(name).get_simulator_requirements(params).items():
                if key in requirements and requirements[key] != value:
                    raise ValueError(f"Conflicting simulator requirement '{key}'.")
                requirements[key] = value
        return requirements


    def matches_config(
        self,
        num_envs: int,
        terms: Mapping[str, Any] | None = None,
    ) -> bool:

        configs = self._term_configs(terms)
        return (
            self.num_envs == num_envs
            and self.terms.keys() == configs.keys()
            and all(self.terms[name].matches_config(params) for name, params in configs.items())
        )


    def bind(
        self,
        backend: RandomizationBackend,
    ) -> None:

        for term in self.terms.values():
            term.bind(backend)
        self._bound = True


    def before_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if not self._bound:
            raise RuntimeError("Bind randomization to a simulator before reset.")
        try:
            for term in self.terms.values():
                term.before_simulator_reset(env_ids)
        except Exception as error:
            self._abort_after_error(env_ids, error)
            raise


    def after_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if not self._bound:
            raise RuntimeError("Bind randomization to a simulator before reset.")
        try:
            for term in self.terms.values():
                term.after_simulator_reset(env_ids)
        except Exception as error:
            self._abort_after_error(env_ids, error)
            raise


    def commit_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        for term in self.terms.values():
            term.commit_simulator_reset(env_ids)


    def _abort_after_error(
        self,
        env_ids: torch.Tensor | None,
        error: Exception,
    ) -> None:

        try:
            self.abort_simulator_reset(env_ids)
        except Exception as rollback_error:
            raise ExceptionGroup("Randomization reset and rollback failed", [error, rollback_error]) from error


    def abort_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        failures = []
        for term in reversed(self.terms.values()):
            try:
                term.abort_simulator_reset(env_ids)
            except Exception as error:
                failures.append(error)
        if failures:
            raise ExceptionGroup("Randomization rollback failed", failures)
