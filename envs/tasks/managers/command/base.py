import torch
from typing import Any
from copy import deepcopy
from functools import partial
from collections.abc import Mapping

from .terms import BaseCommandTerm, get_command_class
from .playback import PlaybackCommandSource, PlaybackCommands
from .constraints import CommandConstraintSet
from .curriculum import curriculum_reference
from envs.tasks.managers.curriculum.types import (
    ContinuousDimensionConfig,
    CurriculumSpaceConfig,
    CurriculumSpaceConfigs,
)
from envs.simulators.utils import ModelContext
from envs.tasks.utils import TaskContext
from envs.tasks.managers.curriculum import (
    CommandCurriculumSampler,
    CurriculumTermProvider,
)
from utils import RuntimeContext


class CommandManager:

    @classmethod
    def get_curriculum_space_config(
        cls,
        config: Mapping[str, Any] | None,
    ) -> CurriculumSpaceConfigs:

        config = config if config is not None else {"terms": {}}
        command_terms = config.get("terms")
        if not isinstance(command_terms, dict):
            raise ValueError("Command manager must define 'terms'.")

        dimensions: dict[str, dict[str, ContinuousDimensionConfig]] = {}
        term_curricula: dict[str, str] = {}
        for term_name, term_config in command_terms.items():

            if not isinstance(term_config, dict):
                raise TypeError(
                    f"Config of command term '{term_name}' must be a dict."
                )

            type_name = term_config.get("type")
            if not isinstance(type_name, str):
                raise ValueError(
                    f"'type' is missing for command term '{term_name}'."
                )

            params = term_config.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise TypeError(
                    f"'params' of command term '{term_name}' must be a dict."
                )

            curriculum_name = curriculum_reference(get_command_class(type_name), params)
            if curriculum_name is None:
                continue

            if "min_value" not in params or "max_value" not in params:
                raise ValueError(f"Command term '{term_name}' requires min_value and max_value.")

            curriculum_dimensions = dimensions.setdefault(curriculum_name, {})
            curriculum_dimensions[term_name] = ContinuousDimensionConfig(
                min_value=params["min_value"],
                max_value=params["max_value"],
                num_cell=params.get("num_cell", 1),
                dim=params.get("dim", 1),
            )
            term_curricula[term_name] = curriculum_name

        constraints = CommandConstraintSet(set(command_terms), config.get("constraints"))
        constraints.validate_curricula(term_curricula)
        return {
            curriculum_name: CurriculumSpaceConfig(
                dimensions=dict(curriculum_dimensions),
                filters=(partial(constraints.filter, target_names=set(curriculum_dimensions)),),
                source_config={"command": deepcopy(dict(config))},
            )
            for curriculum_name, curriculum_dimensions in dimensions.items()
        }


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
        for term_name, term in self.terms.items():
            self.command[term_name] = torch.zeros_like(term.command)
            self.last_command[term_name] = torch.zeros_like(
                self.command[term_name]
            )

        self.output_dim = sum(
            term.dim
            for term in self.terms.values()
        )
        self._playback_commands: PlaybackCommands | None = None


    def activate_playback_commands(
        self,
        source: PlaybackCommandSource,
        env_index: int,
        fixed_commands: Mapping[str, tuple[float, ...]] | None = None,
    ) -> None:

        if self._playback_commands is not None:
            raise RuntimeError("Playback commands are already active.")
        if not 0 <= env_index < self.num_envs:
            raise ValueError("Playback environment index is out of range.")

        playback = PlaybackCommands(
            terms=self.terms,
            source=source,
            env_index=env_index,
            fixed_commands=fixed_commands,
        )
        self.terms.update(playback.terms)
        self._playback_commands = playback
        self._playback_reset_completed = False


    def deactivate_playback_commands(self) -> None:

        playback = self._playback_commands
        if playback is None:
            return

        self.terms.update(playback.original_terms)
        self._playback_commands = None
        if not self._playback_reset_completed:
            return
        env_ids = torch.tensor(
            [playback.env_index],
            dtype=torch.long,
            device=self.context.device
        )
        for term in self.terms.values():
            term.reset(env_ids)
        checked_command = self._constraints_check(self.proposed_command, env_ids=env_ids)
        self._update_command(checked_command, env_ids=env_ids)


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
            params = dict(params)
            term_kwargs = self._get_term_kwargs(cls, name, params, *args, **kwargs)

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
        params: dict[str, Any],
        *args, **kwargs,
    ) -> dict[str, Any]:

        term_kwargs: dict[str, Any] = {}
        curriculum_name = curriculum_reference(cls, params)

        if curriculum_name is not None:

            params.pop("curriculum")
            curriculum_manager = kwargs.pop("curriculum_manager", None)
            if not isinstance(curriculum_manager, CurriculumTermProvider):
                raise ValueError(
                    f"Command term '{name}' requires a "
                    "curriculum term provider."
                )

            curriculum_term = curriculum_manager.get_term(
                curriculum_name
            )
            if not isinstance(curriculum_term, CommandCurriculumSampler):
                raise TypeError(
                    f"Curriculum term '{curriculum_name}' "
                    "does not provide command sampling."
                )

            term_kwargs["curriculum_sampler"] = curriculum_term
            term_kwargs["term_name"] = name

        return term_kwargs | kwargs


    def update(
        self,
        task_context: TaskContext,
    ) -> dict:

        for _, term in self.terms.items():
            term.update(task_context)

        checked_command = self._constraints_check(self.proposed_command)
        self._update_command(checked_command)

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

        checked_command = self._constraints_check(
            self.proposed_command,
            env_ids=env_ids,
        )
        self._update_command(checked_command, env_ids=env_ids)
        if self._playback_commands is not None:
            self._playback_reset_completed = True


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

        checked_command = self.constraint_set.correct(checked_command)
        return checked_command


    def _update_command(
        self,
        checked_command: dict[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:

        for name, value in checked_command.items():
            if env_ids is None:
                self.last_command[name].copy_(
                    self.command[name]
                )
                self.command[name].copy_(value)
            else:
                self.last_command[name][env_ids] = self.command[name][env_ids]
                self.command[name][env_ids] = value
