import torch
from typing import Any

from app.utils.context import RuntimeContext
from envs.simulators.utils.context import ModelContext
from envs.tasks.utils.context import TaskContext
from envs.tasks.managers.command.terms.base import BaseCommandTerm
from envs.tasks.managers.command.terms.registry import get_command_class
from envs.tasks.managers.command.constraints import CommandConstraintSet
from envs.tasks.managers.curriculum.types import (
    CommandCurriculumSampler,
    CurriculumTermProvider,
)


class CommandManager:

    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        model_context: ModelContext,
        terms: dict[str, dict[str, Any]],
        constraints: dict[str, dict[str, Any]] | None = None,
        *args, **kwargs,
    ) -> None:

        self.num_envs = num_envs
        self.context = context
        self.model_context = model_context

        self.terms: dict[str, BaseCommandTerm] = {}
        self._build_terms(terms, *args, **kwargs)
        self.constraint_set = CommandConstraintSet(
            set(self.terms),
            constraints,
        )

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
        *args, **kwargs
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
            term_kwargs = self._get_term_kwargs(cls, name, *args, **kwargs)

            self.terms[name] = cls(
                num_envs=self.num_envs,
                context=self.context,
                model_context=self.model_context,
                **term_kwargs,
                **params,
            )


    def _get_term_kwargs(
        self,
        cls: type[BaseCommandTerm],
        name: str,
        *args, **kwargs,
    ) -> dict[str, Any]:

        term_kwargs: dict[str, Any] = {}
        curriculum_term_name = getattr(cls, "curriculum_term_name", None)

        if curriculum_term_name is not None:

            curriculum_manager = kwargs.get("curriculum_manager")
            if not isinstance(curriculum_manager, CurriculumTermProvider):
                raise ValueError(
                    f"Command term '{name}' requires a "
                    "curriculum term provider."
                )
            
            curriculum_term = curriculum_manager.get_term(
                curriculum_term_name
            )
            if not isinstance(curriculum_term, CommandCurriculumSampler):
                raise TypeError(
                    f"Curriculum term '{curriculum_term_name}' "
                    "does not provide command sampling."
                )
            
            term_kwargs["curriculum_sampler"] = curriculum_term
            term_kwargs["term_name"] = name

        return term_kwargs


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

        checked_command = self.constraint_set.apply(checked_command)

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
