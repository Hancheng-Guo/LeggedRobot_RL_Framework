# envs/tasks/managers/command/base.py

import torch
import re
from collections.abc import Callable
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
        constraints: dict[str, dict[str, Any]] | None = None,
        *args,
        **kwargs,
    ) -> None:

        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.terms: dict[str, BaseCommandTerm] = {}
        self._build_terms(terms)
        self.constraints: dict[
            str,
            Callable[
                [torch.Tensor, dict[str, torch.Tensor]],
                torch.Tensor,
            ],
        ] = {}
        self.constraints = self._build_constraints(constraints)

        self.command: dict[str, torch.Tensor] = {}
        self.last_command: dict[str, torch.Tensor] = {}
        for term_name in self.terms.keys():
            self.command[term_name] = torch.zeros(
                (self.num_envs, 1),
                dtype=self.context.dtype,
                device=self.context.device,
            )
            self.last_command[term_name] = torch.zeros_like(
                self.command[term_name]
            )


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


    def _build_constraints(
        self,
        constraints: dict[str, dict[str, Any]] | None,
    ) -> dict[
        str,
        Callable[
            [torch.Tensor, dict[str, torch.Tensor]],
            torch.Tensor,
        ],
    ]:

        if constraints is None:
            return {}

        built_constraints: dict[
            str,
            Callable[
                [torch.Tensor, dict[str, torch.Tensor]],
                torch.Tensor,
            ],
        ] = {}

        for term_name, constraint_config in constraints.items():

            if term_name not in self.terms:
                raise ValueError(
                    f"Constraint target '{term_name}' is not a command term."
                )

            if not isinstance(constraint_config, dict):
                raise TypeError(
                    f"Config of constraint '{term_name}' must be a dict."
                )

            operator = constraint_config.get("operator")
            expression = constraint_config.get("expression")

            if operator is None or expression is None:
                raise ValueError(
                    f"'operator' or 'expression' is missing for constraint '{term_name}'."
                )

            if operator not in {"<=", ">="}:
                raise ValueError(
                    f"Unsupported operator '{operator}' "
                    f"for constraint '{term_name}'."
                )

            if not isinstance(expression, str):
                raise TypeError(
                    f"'expression' of constraint '{term_name}' must be a string."
                )

            referenced_terms = set(
                re.findall(r"\{([A-Za-z_]\w*)\}", expression)
            )
            unknown_terms = referenced_terms - self.terms.keys()
            if unknown_terms:
                raise ValueError(
                    f"Constraint '{term_name}' references unknown command "
                    f"term(s): {sorted(unknown_terms)}."
                )

            compiled_expression = compile(
                re.sub(
                    r"\{([A-Za-z_]\w*)\}",
                    lambda match: f"commands[{match.group(1)!r}]",
                    expression,
                ),
                f"<command constraint: {term_name}>",
                "eval",
            )

            def apply_constraint(
                command: torch.Tensor,
                command_context: dict[str, torch.Tensor],
                *,
                compiled_expression=compiled_expression,
                constraint_operator=operator,
                referenced_terms=referenced_terms,
                term_name=term_name,
            ) -> torch.Tensor:
                commands = {
                    name: command_context[name]
                    for name in referenced_terms
                }
                try:
                    boundary = eval(
                        compiled_expression,
                        {"__builtins__": {}},
                        {
                            "commands": commands,
                            "abs": torch.abs,
                            "torch": torch,
                        },
                    )
                    boundary = torch.as_tensor(
                        boundary,
                        dtype=command.dtype,
                        device=command.device,
                    )
                except Exception as error:
                    raise ValueError(
                        f"Error evaluating constraint '{term_name}': {error}"
                    ) from error

                if constraint_operator == "<=":
                    return torch.minimum(command, boundary)

                return torch.maximum(command, boundary)

            built_constraints[term_name] = apply_constraint

        return built_constraints


    def update(
        self,
        task_context: TaskContext,
    ) -> dict:

        for _, term in self.terms.items():
            term.update(task_context)

        checked_command = self._constraints_check(self.proposed_command)

        return {
            f"command/{name}": value
            for name, value in checked_command.items()
        }


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        for _, term in self.terms.items():
            term.reset(env_ids)

        self._constraints_check(
            self.proposed_command,
            env_ids=env_ids,
        )


    def get_command(
        self,
        name: str,
    ) -> torch.Tensor:

        return self.command[name]


    @property
    def proposed_command(
        self,
    ) -> dict[str, torch.Tensor]:

        return {
            name: term.command
            for name, term in self.terms.items()
        }


    def _constraints_check(
        self,
        proposed_command: dict[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        if env_ids is None:
            checked_command = dict(proposed_command)
        else:
            checked_command = {
                name: value[env_ids]
                for name, value in proposed_command.items()
            }

        for name, constraint in self.constraints.items():
            checked_command[name] = constraint(
                checked_command[name],
                checked_command,
            )

        for name, value in checked_command.items():
            if env_ids is None:
                self.last_command[name].copy_(
                    self.command[name]
                )
                self.command[name].copy_(value)
            else:
                self.last_command[name][env_ids].copy_(
                    self.command[name][env_ids]
                )
                self.command[name][env_ids].copy_(value)

        return checked_command
