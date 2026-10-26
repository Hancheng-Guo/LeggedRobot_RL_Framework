import torch
from typing import Any
from collections.abc import Mapping

from envs.simulators.utils import ModelContext
from envs.tasks.managers.curriculum.terms import (
    BaseCurriculumTerm,
    get_curriculum_class,
)
from utils import RuntimeContext


class CurriculumManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: dict[str, dict[str, Any] | None],
        manager_configs: dict[str, Any],
        *args, **kwargs,
    ) -> None:
        
        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.terms: dict[str, BaseCurriculumTerm] = {}
        self._build_terms(terms, manager_configs)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any] | None],
        manager_configs: dict[str, Any],
    ) -> None:
        
        if not isinstance(terms, dict):
            raise ValueError("Curriculum 'terms' must be a dict.")

        for name, config in terms.items():

            if config is None:
                config = {}
            elif not isinstance(config, dict):
                raise TypeError(
                    f"Config of curriculum term '{name}' must be a dict."
                )

            cls = get_curriculum_class(name)
            term_kwargs = self._get_term_kwargs(cls, name, manager_configs)

            self.terms[name] = cls(
                num_envs=self.num_envs,
                context=self.context,
                model_context=self.model_context,
                **term_kwargs,
                **config,
            )


    def _get_term_kwargs(
        self,
        cls: type[BaseCurriculumTerm],
        name: str,
        manager_configs: dict[str, Any],
    ) -> dict[str, Any]:
        
        term_kwargs: dict[str, Any] = {}
        config_names = getattr(cls, "manager_config_names", ())

        for config_name in config_names:

            if config_name not in manager_configs:
                raise ValueError(
                    f"Curriculum term '{name}' requires "
                    f"'{config_name}'."
                )
            
            term_kwargs[config_name] = manager_configs[config_name]

        return term_kwargs


    def get_term(
        self,
        name: str
    ) -> BaseCurriculumTerm:

        if name not in self.terms:
            raise KeyError(f"Unknown curriculum term '{name}'.")
        
        return self.terms[name]


    def update(
        self,
        *args, **kwargs
    ) -> dict[str, torch.Tensor]:
        
        info: dict[str, torch.Tensor] = {}
        for name, term in self.terms.items():
            term_info = term.update(*args, **kwargs)
            info.update({
                f"curriculum/{name}/{key}": value
                for key, value in term_info.items()
            })

        return info


    def collect_metrics(self) -> dict[str, torch.Tensor]:
        info: dict[str, torch.Tensor] = {}
        for name, term in self.terms.items():
            info.update({
                f"curriculum/{name}/{key}": value
                for key, value in term.collect_metrics().items()
            })
        return info


    def get_metric_names(self) -> set[str]:
        return {
            f"curriculum/{name}/{metric}"
            for name, term in self.terms.items()
            for metric in term.get_metric_names()
        }


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        for term in self.terms.values():
            term.reset(env_ids)


    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "terms": {
                name: term.checkpoint_state_dict()
                for name, term in self.terms.items()
            },
        }


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:

        if not isinstance(state, Mapping):
            raise TypeError(
                "Curriculum manager checkpoint must be a mapping."
            )

        terms = state.get("terms")
        if (
            not isinstance(terms, Mapping)
            or set(terms) != set(self.terms)
        ):
            raise ValueError(
                "Checkpoint curriculum terms must match configured terms."
            )

        # Reject incompatible states before restoring any term.
        for name, term in self.terms.items():
            term.validate_checkpoint_state_dict(terms[name])
        for name, term in self.terms.items():
            term.load_checkpoint_state_dict(terms[name])
