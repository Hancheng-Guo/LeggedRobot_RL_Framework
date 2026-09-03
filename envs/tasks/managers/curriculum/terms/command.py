import torch
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from envs.tasks.managers.command.constraints import CommandConstraintSet
from envs.tasks.managers.curriculum.terms.base import BaseCurriculumTerm
from envs.tasks.managers.curriculum.terms.registry import register_curriculum


@dataclass
class CommandCurriculumBuffer:
    dimension_names: tuple[str, ...]
    term_slices: dict[str, slice]
    command_values: torch.Tensor
    reward_sum: torch.Tensor
    sample_count: torch.Tensor
    assigned_cell_ids: torch.Tensor


class CommandRewardCurriculum(BaseCurriculumTerm, ABC):

    manager_config_names = ("command_manager_config",)

    def __init__(
        self,
        command_manager_config: dict[str, Any],
        command_term_type: str,
        temperature: float = 100.0,
        exploration: float = 0.1,
        max_cells: int = 100_000,
        *args, **kwargs,
    ) -> None:
        
        super().__init__(*args, **kwargs)

        if temperature <= 0.0:
            raise ValueError("'temperature' must be greater than 0.")
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("'exploration' must be between 0 and 1.")

        self.command_term_type = command_term_type
        self.temperature = temperature
        self.exploration = exploration
        self.max_cells = max_cells

        self.spaces = self._extract_spaces(command_manager_config)
        self.constraint_set = self._extract_constraint_set(
            command_manager_config,
            self.spaces
        )
        self.buffers = self._build_buffer()
        

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
            if term_config.get("type") != self.command_term_type:
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
                "dim": params.get("dim", 1),
            }

        if not spaces:
            raise ValueError(
                f"No {self.command_term_type} terms were found in the "
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


    def _build_buffer(self) -> dict[str, CommandCurriculumBuffer]:
        return {
            name: self._build_buffer_term(
                name,
                dimensions,
                self.constraint_set,
            )
            for name, dimensions in self.spaces.items()
        }


    def _build_buffer_term(
        self,
        name: str,
        dimensions: dict[str, dict[str, Any]],
        constraint_set: CommandConstraintSet,
    ) -> CommandCurriculumBuffer:

        # get the axes for each dimension and validate the configuration
        axes: list[torch.Tensor] = []
        dimension_names: list[str] = []
        term_slices: dict[str, slice] = {}
        raw_num_cells = 1
        for term_name, config in dimensions.items():

            num_bins = int(config.get("num_bins") or 0)
            if num_bins <= 0:
                raise ValueError(
                    f"'num_bins' of command term '{term_name}' "
                    "must be positive."
                )

            command_dim = config.get("dim", 1)
            if (
                not isinstance(command_dim, int)
                or isinstance(command_dim, bool)
                or command_dim <= 0
            ):
                raise ValueError(
                    f"'command_dim' of command term '{term_name}' must be "
                    "a positive integer."
                )

            raw_num_cells *= num_bins ** command_dim
            if raw_num_cells > self.max_cells:
                raise ValueError(
                    f"Curriculum space '{name}' has at least "
                    f"{raw_num_cells} raw cells, exceeding "
                    f"max_cells={self.max_cells}. Reduce 'num_bins' or "
                    "'command_dim', or split the command terms into "
                    "different groups."
                )
            
            min_value = float(config["min_value"])
            max_value = float(config["max_value"])
            if min_value > max_value:
                raise ValueError(
                    f"'min_value' of command term '{term_name}' "
                    "cannot exceed 'max_value'."
                )

            axis = torch.linspace(
                min_value,
                max_value,
                num_bins,
                dtype=self.context.dtype,
                device=self.context.device,
            )
            start = len(axes)
            axes.extend([axis] * command_dim)
            term_slices[term_name] = slice(start, start + command_dim)
            dimension_names.extend(
                [term_name]
                if command_dim == 1
                else [
                    f"{term_name}[{index}]"
                    for index in range(command_dim)
                ]
            )

        # compute command value of each cell
        all_command_values = (
            axes[0].unsqueeze(-1)
            if len(axes) == 1
            else torch.cartesian_prod(*axes)
        )
        all_commands = {
            term_name: all_command_values[:, term_slice]
            for term_name, term_slice in term_slices.items()
        }

        # apply constraints to command values
        checked_commands = constraint_set.apply(
            all_commands,
            target_names=set(dimensions),
        )
        checked_command_values = torch.cat(
            [checked_commands[name] for name in dimensions],
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
            dimension_names=tuple(dimension_names),
            term_slices=term_slices,
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
            term_slice = buffer.term_slices[dimension]
        except KeyError as error:
            raise ValueError(
                f"Unknown command term '{dimension}' in curriculum "
                f"space '{space_name}'."
            ) from error
        
        cell_ids = buffer.assigned_cell_ids
        if env_ids is not None:
            cell_ids = cell_ids[env_ids]
        if torch.any(cell_ids < 0):
            raise RuntimeError(
                f"Curriculum space '{space_name}' has not been sampled."
            )
        
        return buffer.command_values[cell_ids, term_slice]


    @abstractmethod
    def update(     # update rewards to cell buffer
        self,
        reward: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Subclasses of CommandRewardCurriculum must implement "
            "update() to update the reward and sample count of each cell."
        )


    @abstractmethod
    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:
        raise NotImplementedError(
            "Subclasses of CommandRewardCurriculum must implement "
            "reset() to reset the assigned cell IDs for each environment."
        )

    
    @abstractmethod
    def resample(
        self,
        space_names: set[str],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError(
            "Subclasses of CommandRewardCurriculum must implement "
            "resample() to sample new cell IDs for each environment."
        )


    @abstractmethod
    def _get_probabilities(
        self,
        space_name: str
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Subclasses of CommandRewardCurriculum must implement "
            "_get_probabilities() to compute the probabilities of each cell."
        )


@register_curriculum
class LrpcCommandReward(CommandRewardCurriculum):

    def __init__(
        self,
        *args, **kwargs,
    ) -> None:

        super().__init__(
            command_term_type="LrpcSampleOnReset",
            *args, **kwargs
        )


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

            info[f"{name}/probabilities"] = self._get_probabilities(name)

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
                self._get_probabilities(space_name),    # propabilities of each cell
                selected_count,                         # number of cells to sample
                replacement=True,                       # allow duplicates
            )

            if env_ids is None:
                buffer.assigned_cell_ids.copy_(sampled_ids)
            else:
                buffer.assigned_cell_ids[env_ids] = sampled_ids


    def _get_probabilities(
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

        return (
            (1.0 - self.exploration) * probabilities
            + self.exploration / probabilities.numel()
        )



@register_curriculum
class LpacCommandReward(CommandRewardCurriculum):

    def __init__(
        self,
        min_coverage: float = 0.8,
        min_samples_per_cell: int = 5,
        *args, **kwargs,
    ) -> None:

        super().__init__(
            command_term_type="LpacSampleOnReset",
            *args, **kwargs
        )

        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("'min_coverage' must be in (0, 1].")
        if (
            not isinstance(min_samples_per_cell, int)
            or isinstance(min_samples_per_cell, bool)
            or min_samples_per_cell <= 0
        ):
            raise ValueError(
                "'min_samples_per_cell' must be a positive integer."
            )

        self.min_coverage = min_coverage
        self.min_samples_per_cell = min_samples_per_cell

        self.last_reward_mean = {
            name: torch.zeros_like(buffer.reward_sum)
            for name, buffer in self.buffers.items()
        }
        self.has_previous = {
            name: torch.zeros_like(buffer.sample_count, dtype=torch.bool)
            for name, buffer in self.buffers.items()
        }
        self.learning_progress = {
            name: torch.zeros_like(buffer.reward_sum)
            for name, buffer in self.buffers.items()
        }
        self.probabilities = {
            name: self._calculate_probabilities(name)
            for name in self.buffers
        }
        self.episode_reward_sum = torch.zeros(
            self.num_envs,
            dtype=self.context.dtype,
            device=self.context.device,
        )
        self.episode_step_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.context.device,
        )


    def update(     # update rewards to cell buffer
        self,
        reward: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        self.episode_reward_sum += reward
        self.episode_step_count += 1

        return {
            f"{name}/learning_progress": learning_progress
            for name, learning_progress in self.learning_progress.items()
        }


    def reset(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        collected = self.episode_step_count > 0
        valid = torch.ones(self.num_envs,dtype=torch.bool,device=self.context.device)
        for buffer in self.buffers.values():
            valid &= buffer.assigned_cell_ids >= 0
        if env_ids is None:
            done = torch.arange(
                self.num_envs,
                dtype=torch.long,
                device=self.context.device,
            )[(collected & valid)]
        else:
            done = env_ids[(collected & valid)[env_ids]]

        for name, buffer in self.buffers.items():
            cell_ids = buffer.assigned_cell_ids[done]
            buffer.reward_sum.scatter_add_(0, cell_ids, self.episode_reward_sum[done])
            buffer.sample_count.scatter_add_(0, cell_ids, torch.ones_like(cell_ids))
            buffer.assigned_cell_ids[done] = -1

            count_satisfied = (
                buffer.sample_count >= self.min_samples_per_cell
            )
            coverage = (
                torch.count_nonzero(count_satisfied).item()
                / buffer.sample_count.numel()
            )
            if coverage >= self.min_coverage:
                reward_mean = self._calculate_reward_mean(name)
                has_previous = self.has_previous[name]
                first_window = count_satisfied & ~has_previous
                later_window = count_satisfied & has_previous

                self.learning_progress[name][first_window] = 0.0
                self.learning_progress[name][later_window] = (
                    reward_mean[later_window]
                    - self.last_reward_mean[name][later_window]
                )
                self.last_reward_mean[name][count_satisfied] = (
                    reward_mean[count_satisfied]
                )
                has_previous[count_satisfied] = True
                self.probabilities[name] = self._calculate_probabilities(name)

                buffer.sample_count[count_satisfied] = 0
                buffer.reward_sum[count_satisfied] = 0

        if env_ids is None:
            self.episode_step_count.fill_(0)
            self.episode_reward_sum.fill_(0)
        else:
            self.episode_step_count[done] = 0
            self.episode_reward_sum[done] = 0

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
                self._get_probabilities(space_name),    # propabilities of each cell
                selected_count,                         # number of cells to sample
                replacement=True,                       # allow duplicates
            )

            if env_ids is None:
                buffer.assigned_cell_ids.copy_(sampled_ids)
            else:
                buffer.assigned_cell_ids[env_ids] = sampled_ids


    def _get_probabilities(
        self,
        space_name: str
    ) -> torch.Tensor:
        
        return self.probabilities[space_name]


    def _calculate_reward_mean(
        self,
        space_name: str
    ) -> torch.Tensor:

        buffer = self.buffers[space_name]
        counts = buffer.sample_count.clamp_min(1).to(self.context.dtype)
        reward_mean = buffer.reward_sum / counts

        return reward_mean


    def _calculate_probabilities(
        self,
        space_name: str
    ) -> torch.Tensor:

        probabilities = torch.softmax(
            self.learning_progress[space_name] / self.temperature,
            dim=0,
        )

        return (
            (1.0 - self.exploration) * probabilities
            + self.exploration / probabilities.numel()
        )
