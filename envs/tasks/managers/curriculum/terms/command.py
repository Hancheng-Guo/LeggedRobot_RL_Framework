import torch
from dataclasses import dataclass
from typing import Any

from envs.tasks.managers.command.constraints import CommandConstraintSet
from envs.tasks.managers.curriculum.terms.base import BaseCurriculumTerm
from envs.tasks.managers.curriculum.terms.registry import register_curriculum


@dataclass
class CommandCurriculumBuffer:
    dimension_names: tuple[str, ...]
    command_values: torch.Tensor
    reward_sum: torch.Tensor
    sample_count: torch.Tensor
    assigned_cell_ids: torch.Tensor


@register_curriculum
class CommandReward(BaseCurriculumTerm):

    manager_config_names = ("command_manager_config",)

    def __init__(
        self,
        command_manager_config: dict[str, Any],
        temperature: float = 1.0,
        exploration: float = 0.05,
        max_cells: int = 100_000,
        *args,
        **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if temperature <= 0.0:
            raise ValueError("'temperature' must be greater than 0.")
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("'exploration' must be between 0 and 1.")

        self.temperature = temperature
        self.exploration = exploration
        self.max_cells = max_cells

        spaces = self._extract_spaces(command_manager_config)
        constraint_set = self._extract_constraint_set(
            command_manager_config,
            spaces
        )
        self.buffers = {
            name: self._build_buffer(
                name,
                dimensions,
                constraint_set,
            )
            for name, dimensions in spaces.items()
        }


    def _extract_spaces(
        self,
        command_manager_config: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        
        command_terms = command_manager_config.get("terms")
        if not isinstance(command_terms, dict):
            raise ValueError("Command manager must define 'terms'.")

        spaces: dict[str, dict[str, Any]] = {}
        for term_name, term_config in command_terms.items():

            if not isinstance(term_config, dict):
                raise TypeError(
                    f"Config of command term '{term_name}' must be a dict."
                )
            if term_config.get("type") != "CurriculumSampleOnReset":
                continue

            params = term_config.get("params", {})
            if not isinstance(params, dict):
                raise TypeError(
                    f"'params' of command term '{term_name}' must be a dict."
                )
            
            group = params.get("group", None)
            if group is None:
                raise ValueError(
                    f"Command term '{term_name}' requires a 'group' parameter."
                )
            if not isinstance(group, str):
                raise TypeError(
                    f"'group' of command term '{term_name}' must be a string."
                )
            
            spaces.setdefault(group, {})[term_name] = {
                "min_value": params.get("min_value"),
                "max_value": params.get("max_value"),
                "num_bins": params.get("num_bins"),
            }

        if not spaces:
            raise ValueError(
                "No CurriculumSampleOnReset terms were found in the "
                "command configuration."
            )
        
        return spaces


    def _extract_constraint_set(
        self,
        command_manager_config: dict[str, Any],
        spaces: dict[str, dict[str, Any]],
    ) -> CommandConstraintSet:

        command_terms = command_manager_config["terms"]
        constraint_set = CommandConstraintSet(
            set(command_terms),
            command_manager_config.get("constraints"),
        )
        term_groups = {
            term_name: group
            for group, dimensions in spaces.items()
            for term_name in dimensions
        }
        constraint_set.validate_groups(term_groups)

        return constraint_set


    def _build_buffer(
        self,
        name: str,
        dimensions: dict[str, dict[str, Any]],
        constraint_set: CommandConstraintSet,
    ) -> CommandCurriculumBuffer:

        # get the axes for each dimension and validate the configuration
        axes: list[torch.Tensor] = []
        for dimension, config in dimensions.items():

            num_bins = int(config.get("num_bins") or 0)
            if num_bins <= 0:
                raise ValueError(
                    f"'num_bins' of dimension '{dimension}' must be positive."
                )
            
            min_value = float(config["min_value"])
            max_value = float(config["max_value"])
            if min_value > max_value:
                raise ValueError(
                    f"'min_value' of dimension '{dimension}' cannot exceed "
                    "'max_value'."
                )
            
            axes.append(torch.linspace(
                min_value,
                max_value,
                num_bins,
                dtype=self.context.dtype,
                device=self.context.device,
            ))

        # compute command value of each cell
        all_command_values = (
            axes[0].unsqueeze(-1)
            if len(axes) == 1
            else torch.cartesian_prod(*axes)
        )
        dimension_names = tuple(dimensions)
        all_commands = {
            dimension: all_command_values[:, index:index + 1]
            for index, dimension in enumerate(dimension_names)
        }

        # apply constraints to command values
        checked_commands = constraint_set.apply(
            all_commands,
            target_names=set(dimension_names),
        )
        checked_command_values = torch.cat(
            [checked_commands[name] for name in dimension_names],
            dim=-1,
        )

        # filter out any cells that violate constraints
        filtered_command_values = self._filter_command_values(
            all_command_values,
            checked_command_values,
        )

        # check if the number of cells exceeds max_cells
        num_cells = filtered_command_values.shape[0]
        if num_cells > self.max_cells:
            raise ValueError(
                f"Curriculum space '{name}' has {num_cells} cells, "
                f"exceeding max_cells={self.max_cells}."
            )

        return CommandCurriculumBuffer(
            dimension_names=dimension_names,
            command_values=filtered_command_values,
            reward_sum=torch.zeros(
                num_cells,
                dtype=self.context.dtype,
                device=self.context.device,
            ),
            sample_count=torch.zeros(
                num_cells,
                dtype=torch.long,
                device=self.context.device,
            ),
            assigned_cell_ids=torch.full(
                (self.num_envs,),
                -1,
                dtype=torch.long,
                device=self.context.device,
            ),
        )


    def _filter_command_values(
        self,
        all_command_values: torch.Tensor,
        checked_command_values: torch.Tensor,
    ) -> torch.Tensor:

        combined_values = torch.cat(
            (all_command_values, checked_command_values),
            dim=0,
        )
        _, inverse_ids = torch.unique(
            combined_values,
            dim=0,
            return_inverse=True,
        )

        num_all_values = all_command_values.shape[0]
        all_value_ids = inverse_ids[:num_all_values]
        checked_value_ids = inverse_ids[num_all_values:]
        valid = torch.isin(checked_value_ids, all_value_ids)

        filtered_command_values = checked_command_values[valid].unique(dim=0)
        if filtered_command_values.shape[0] == 0:
            raise ValueError(
                "No command values remain after applying constraints."
            )

        return filtered_command_values


    def get_command(
        self,
        space_name: str,
        dimension: str,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        buffer = self.buffers[space_name]

        try:
            dimension_id = buffer.dimension_names.index(dimension)
        except ValueError as error:
            raise ValueError(
                f"Unknown dimension '{dimension}' in curriculum "
                f"space '{space_name}'."
            ) from error
        
        cell_ids = buffer.assigned_cell_ids
        if env_ids is not None:
            cell_ids = cell_ids[env_ids]
        if torch.any(cell_ids < 0):
            raise RuntimeError(
                f"Curriculum space '{space_name}' has not been sampled."
            )
        
        return buffer.command_values[cell_ids, dimension_id:dimension_id + 1]


    def update(     # update rewards to cell buffer
        self,
        reward: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        info: dict[str, torch.Tensor] = {}

        for name, buffer in self.buffers.items():

            valid = buffer.assigned_cell_ids >= 0
            cell_ids = buffer.assigned_cell_ids[valid]

            buffer.reward_sum.scatter_add_(0, cell_ids, reward[valid])
            buffer.sample_count.scatter_add_(0, cell_ids, torch.ones_like(cell_ids))

            info[f"curriculum/{name}/probabilities"] = self.probabilities(name)

        return info


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        
        for buffer in self.buffers.values():

            if env_ids is None:
                buffer.assigned_cell_ids.fill_(-1)
            else:
                buffer.assigned_cell_ids[env_ids] = -1

        self.resample(set(self.buffers), env_ids)


    def resample(
        self,
        space_names: set[str],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        
        selected_count = self.num_envs if env_ids is None else env_ids.numel()

        for space_name in space_names:

            buffer = self.buffers[space_name]
            sampled_ids = torch.multinomial(
                self.probabilities(space_name), # propabilities of each cell
                selected_count,                 # number of cells to sample
                replacement=True,               # allow duplicates
            )

            if env_ids is None:
                buffer.assigned_cell_ids.copy_(sampled_ids)
            else:
                buffer.assigned_cell_ids[env_ids] = sampled_ids


    def probabilities(
        self,
        space_name: str
    ) -> torch.Tensor:
        
        buffer = self.buffers[space_name]

        # avoid division by zero by clamping sample counts to at least 1
        counts = buffer.sample_count.clamp_min(1).to(self.context.dtype)

        reward_mean = buffer.reward_sum / counts
        probabilities = torch.softmax(
            -reward_mean / self.temperature,
            dim=0,
        )

        # (1 - exploration) * curriculum_probabilities
        # + exploration * uniform_probabilities
        return (
            (1.0 - self.exploration) * probabilities
            + self.exploration / probabilities.numel()
        )
