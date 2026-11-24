import torch
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

from . import SimulatorState
from .terms import BaseSimulatorStateTerm, get_simulator_state_term_class
from ..utils import ModelContext
from utils import RuntimeContext


class SimulatorStateManager:

    @classmethod
    def validate_config(
        cls,
        *,
        unavailable_capabilities: Mapping[str, str],
        terms: Mapping[str, object] | None = None,
    ) -> None:
        for name, term_class, _ in cls._term_configs(terms):
            required = term_class.required_capabilities
            unavailable = required.intersection(unavailable_capabilities)
            if unavailable:
                reasons = " ".join(
                    unavailable_capabilities[key] for key in sorted(unavailable)
                )
                raise ValueError(
                    f"Simulator state term '{name}' cannot be enabled. {reasons}"
                )


    @staticmethod
    def _term_configs(
        terms: Mapping[str, object] | None,
    ) -> Iterator[tuple[str, type[BaseSimulatorStateTerm], Mapping[str, object]]]:
        if terms is not None and not isinstance(terms, Mapping):
            raise TypeError("Simulator state 'terms' must be a mapping.")
        for name, params in (terms or {}).items():
            if params is not None and not isinstance(params, Mapping):
                raise TypeError(f"Simulator state term '{name}' must be a mapping.")
            yield name, get_simulator_state_term_class(name), params or {}


    def __init__(
        self,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: Mapping[str, object] | None = None,
        **term_dependencies: Any,
    ) -> None:

        self.terms: dict[str, BaseSimulatorStateTerm] = {}
        self._build_terms(context, model_context, terms, **term_dependencies)
        self.model_context = replace(
            model_context,
            state_term_output_shapes={
                name: dict(term.output_shapes)
                for name, term in self.terms.items()
            },
        )


    def _build_terms(
        self,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: Mapping[str, object] | None,
        **term_dependencies: Any,
    ) -> None:
        output_names: set[str] = set()
        for name, term_class, params in self._term_configs(terms):
            term = term_class(
                name, context=context, model_context=model_context,
                **params, **term_dependencies,
            )
            duplicates = output_names & term.output_shapes.keys()
            if duplicates:
                raise ValueError(
                    f"Simulator state term '{name}' declares "
                    f"duplicate fields {sorted(duplicates)}."
                )
            output_names.update(term.output_shapes)
            self.terms[name] = term


    def assemble(
        self,
        env_ids: torch.Tensor | None,
        **state_fields: torch.Tensor,
    ) -> SimulatorState:
        state = SimulatorState(**state_fields)
        term_values: dict[str, torch.Tensor | None] = {}
        for term in self.terms.values():
            term_value = term.compute(state, env_ids)
            term_values.update(term_value)
        state.term_values = term_values
        return state
