import torch
from typing import Any

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from envs.tasks.managers.observation.terms.registry import get_observation_class
from envs.tasks.utils.context import TaskContext


class ObservationManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: dict[str, dict[str, Any] | None],
        clip: float | None = None,
        *args,
        **kwargs,
    ) -> None:

        if clip is not None and clip <= 0.0:
            raise ValueError("'clip' must be greater than 0.")

        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context
        self.clip = clip
        self.terms: dict[str, BaseObservationTerm] = {}
        self._build_terms(terms)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any] | None],
    ) -> None:

        if not isinstance(terms, dict) or not terms:
            raise ValueError("Observation 'terms' must be a non-empty dict.")

        for name, config in terms.items():
            if config is None:
                config = {}
            elif not isinstance(config, dict):
                raise TypeError(
                    f"Config of observation term '{name}' must be a dict."
                )

            cls = get_observation_class(name)
            self.terms[name] = cls(
                context=self.context,
                model_context=self.model_context,
                **config,
            )


    def compute(
        self,
        task_context: TaskContext,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        term_values: list[torch.Tensor] = []
        info: dict[str, torch.Tensor] = {}
        num_selected_envs = (
            self.num_envs
            if env_ids is None
            else env_ids.numel()
        )

        for name, term in self.terms.items():
            value = term.compute(task_context)

            if value.ndim != 2:
                raise ValueError(
                    f"Observation term '{name}' must return a 2D tensor, "
                    f"got shape {tuple(value.shape)}."
                )

            if value.shape[0] != num_selected_envs:
                raise ValueError(
                    f"Observation term '{name}' returned envs size "
                    f"{value.shape[0]}, expected {num_selected_envs}."
                )

            value = value * term.scale
            term_values.append(value)
            info[f"observation/{name}"] = value

        observation = torch.cat(term_values, dim=-1)

        if self.clip is not None:
            observation = observation.clamp(
                min=-self.clip,
                max=self.clip,
            )

        return observation, info


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        
        for term in self.terms.values():
            term.reset(env_ids)
