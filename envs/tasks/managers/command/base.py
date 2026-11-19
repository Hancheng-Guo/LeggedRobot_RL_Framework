import torch
from typing import Any
from copy import deepcopy
from functools import partial
from collections.abc import Mapping

from .terms import BaseCommandTerm, get_command_class
from .playback import PlaybackCommandSource, PlaybackCommands
from .constraints import CommandConstraintSet
from envs.simulators.utils import ModelContext
from envs.tasks.managers.curriculum import (
    CurriculumSpaceConfig,
    CurriculumSpaceConfigs,
)
from envs.tasks.utils import TaskContext, StateManager
from utils import RuntimeContext


class CommandManager(StateManager):

    RESET_STATE_FIELDS = ('command', 'last_command', '_playback_reset_completed')

    @classmethod
    def get_curriculum_space_config(
        cls,
        config: Mapping[str, Any] | None,
    ) -> CurriculumSpaceConfigs:

        config = config if config is not None else {"terms": {}}
        command_terms = config.get("terms")
        if not isinstance(command_terms, dict):
            raise ValueError("Command manager must define 'terms'.")

        spaces: CurriculumSpaceConfigs = {}
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

            term_cls = get_command_class(type_name)
            descriptions = term_cls.get_curriculum_space_config(term_name, params)
            for curriculum_name, space in descriptions.items():
                previous = spaces.get(curriculum_name, CurriculumSpaceConfig({}))
                if previous.dimensions.keys() & space.dimensions.keys():
                    raise ValueError(
                        "Command terms contribute duplicate curriculum dimensions."
                    )
                spaces[curriculum_name] = CurriculumSpaceConfig(
                    dimensions=previous.dimensions | space.dimensions,
                    filters=previous.filters + space.filters,
                    persistent_assignments=(
                        previous.persistent_assignments or space.persistent_assignments
                    ),
                )
                term_curricula[term_name] = curriculum_name

        constraints = CommandConstraintSet(set(command_terms), config.get("constraints"))
        constraints.validate_curricula(term_curricula)
        return {
            curriculum_name: CurriculumSpaceConfig(
                dimensions=space.dimensions,
                filters=space.filters + (
                    partial(constraints.filter, target_names=set(space.dimensions)),
                ),
                persistent_assignments=space.persistent_assignments,
                source_config={"command": deepcopy(dict(config))},
            )
            for curriculum_name, space in spaces.items()
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
        self._playback_reset_completed = False


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
        *args, **kwargs,
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
            self.terms[name] = cls(
                name=name,
                num_envs=self.num_envs,
                context=self.context,
                model_context=self.model_context,
                **kwargs,
                **params,
            )


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
