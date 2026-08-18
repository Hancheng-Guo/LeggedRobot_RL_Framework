# envs/tasks/managers/command/base.py

import torch
from typing import Any

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.base import TaskContext
from envs.tasks.managers.command.terms.base import BaseCommandTerm
from envs.tasks.managers.command.terms.registry import get_command_class


class CommandManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: dict[str, dict[str, Any]],
        *args,
        **kwargs,
    ) -> None:

        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.terms: dict[str, BaseCommandTerm] = {}

        self._build_terms(terms)


    def _build_terms(
        self,
        terms: dict[str, dict[str, Any]],
    ) -> None:

        for name, config in terms.items():

            if name in self.terms:
                raise ValueError(
                    f"Command term '{name}' already exists."
                )

            if not isinstance(config, dict):
                raise TypeError(
                    f"Config of command term '{name}' must be a dict."
                )

            type_name = config.get("type")

            if type_name is None:
                raise ValueError(
                    f"'type' is missing for command term '{name}'."
                )

            params = config.get("params", {})

            if params is None:
                params = {}

            if not isinstance(params, dict):
                raise TypeError(
                    f"'params' of command term '{name}' must be a dict."
                )

            cls = get_command_class(type_name)

            self.terms[name] = cls(
                num_envs=self.num_envs,
                context=self.context,
                model_context=self.model_context,
                **params,
            )


    def update(
        self,
        task_context: TaskContext,
    ) -> dict:

        info = {}

        for name, term in self.terms.items():
            term.update(task_context)
            info[f"command/{name}"] = term.command

        return info


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        for term in self.terms.values():
            term.reset(
                env_ids=env_ids,
            )


    def get_command(
        self,
        name: str,
    ) -> torch.Tensor:

        return self.terms[name].command


    @property
    def command(
        self,
    ) -> dict[str, torch.Tensor]:

        return {
            name: term.command
            for name, term in self.terms.items()
        }


    @property
    def last_command(
        self,
    ) -> dict[str, torch.Tensor]:

        return {
            name: term.last_command
            for name, term in self.terms.items()
        }


