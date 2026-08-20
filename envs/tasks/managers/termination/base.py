from typing import Any

import torch

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.termination.terms.base import BaseTerminationTerm
from envs.tasks.managers.termination.terms.registry import (
    get_termination_class,
)
from envs.tasks.utils.context import TaskContext


class TerminationManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: dict[str, dict[str, Any] | None],
        *args,
        **kwargs,
    ) -> None:
        
        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.terms: dict[str, BaseTerminationTerm] = {}
        self.term_terminated: dict[str, torch.Tensor] = {}
        self._build_terms(terms)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any] | None],
    ) -> None:

        if not isinstance(terms, dict):
            raise TypeError("Termination 'terms' must be a dict.")

        for name, config in terms.items():

            if name in self.terms:
                raise ValueError(f"Termination term '{name}' already exists.")
            
            if config is None:
                config = {}
            elif not isinstance(config, dict):
                raise TypeError(
                    f"Config of termination term '{name}' must be a dict."
                )

            cls = get_termination_class(name)
            self.terms[name] = cls(
                context=self.context,
                model_context=self.model_context,
                **config,
            )
            self.term_terminated[name] = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.context.device,
            )


    def compute(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        terminated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.context.device,
        )
        info: dict[str, torch.Tensor] = {}

        for name, term in self.terms.items():
            
            value = term.compute(task_context)
            if value.dtype != torch.bool:
                raise TypeError(
                    f"Termination term '{name}' must return a bool tensor."
                )
            if value.shape != (self.num_envs,):
                raise ValueError(
                    f"Termination term '{name}' must return shape "
                    f"({self.num_envs},), got {tuple(value.shape)}."
                )

            self.term_terminated[name].copy_(value)
            terminated |= value
            info[f"termination/{name}"] = value

        return terminated, info


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        
        for term in self.terms.values():
            term.reset(env_ids)

        for value in self.term_terminated.values():

            if env_ids is None:
                value.zero_()
            else:
                value[env_ids] = False
