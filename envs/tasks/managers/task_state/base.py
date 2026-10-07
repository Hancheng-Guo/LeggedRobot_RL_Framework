import torch
from typing import Any

from .terms import BaseTaskStateTerm, get_task_state_class
from envs.tasks.utils import TaskContext
from utils import RuntimeContext


class TaskStateManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        terms: dict[str, dict[str, Any] | None] | None = None,
    ) -> None:
        
        self.terms: dict[str, BaseTaskStateTerm] = {}
        for name, params in (terms or {}).items():
            if params is not None and not isinstance(params, dict):
                raise TypeError(f"Config of task state term '{name}' must be a dict.")
            term_class = get_task_state_class(name)
            self.terms[name] = term_class(
                num_envs=num_envs,
                context=context,
                **(params or {}),
            )


    def update(
        self,
        task_context: TaskContext
    ) -> None:
        
        for term in self.terms.values():
            term.update(task_context)
        if task_context.env_ids is not None:
            task_context.task_state = self.values(task_context.env_ids)


    def values(
        self,
        env_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        
        return {
            name: term.value if env_ids is None else term.value[env_ids]
            for name, term in self.terms.items()
        }


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        for term in self.terms.values():
            term.reset(env_ids)
