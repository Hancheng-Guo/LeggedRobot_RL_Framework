import torch
from typing import Any

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.managers.action.terms.base import BaseActionTerm
from envs.tasks.managers.action.terms.registry import get_action_class


class ActionManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: dict[str, dict[str, Any]],
        *args, **kwargs,
    ) -> None:
        
        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.input_dim: int
        self.output_dim = self.model_context.nu
        self.terms: dict[str, BaseActionTerm] = {}

        self.input_dim = self._build_terms(
            terms=terms,
            output_dim=self.output_dim,
        )

        self.action = torch.zeros(
            (self.num_envs, self.input_dim),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.last_action = torch.zeros_like(self.action)

        self.control = torch.zeros(
            (num_envs, self.output_dim),
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.last_control = torch.zeros_like(self.control)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any]],
        output_dim: int,
    ) -> int:

        dim = output_dim

        for name, params in reversed(terms.items()):

            if name in self.terms:
                raise ValueError(
                    f"Action term '{name}' already exists."
                )

            cls = get_action_class(name)

            if params is None or not isinstance(params, dict):
                params = {}

            self.terms[name] = cls(
                output_dim=dim,
                model_context=self.model_context,
                **params
            )

            dim = self.terms[name].input_dim

        return dim


    def process(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        info = {}

        self.last_action.copy_(self.action)
        self.action.copy_(action)

        value = self.action
        for name, term in reversed(self.terms.items()):
            value = term.process(value)
            info[f"action/{name}"] = value

        self.last_control.copy_(self.control)
        self.control.copy_(value)

        info |= {
            "action/action": self.action,
            "action/last_action": self.last_action,
            "action/control": self.control,
            "action/last_control": self.last_control,
        }

        return self.control, info


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        for term in self.terms.values():
            term.reset(env_ids)
        
        if env_ids is None:
            self.action.zero_()
            self.last_action.zero_()
            self.control.zero_()
            self.last_control.zero_()
            return

        self.action[env_ids] = 0
        self.last_action[env_ids] = 0
        self.control[env_ids] = 0
        self.last_control[env_ids] = 0