import torch
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from envs.tasks.managers.command.constraints import CommandConstraintSet
from envs.tasks.managers.command.terms.registry import get_command_class
from envs.tasks.managers.curriculum.terms.base import BaseCurriculumTerm
from envs.tasks.managers.curriculum.terms.registry import register_curriculum


@dataclass
class CommandCurriculumBuffer:
    dimension_names: tuple[str, ...]
    term_slices: dict[str, slice]
    cell_starts: torch.Tensor
    cell_interval: torch.Tensor
    reward_sum: torch.Tensor
    sample_count: torch.Tensor
    assigned_cell_ids: torch.Tensor


class CommandRewardCurriculum(BaseCurriculumTerm, ABC):

    manager_config_names = ("command_manager_config",)

    def __init__(
        self,
        command_manager_config: dict[str, Any],
        command_term_type: str,
        temperature: float = 0.01,
        exploration: float = 0.1,
        max_cells: int = 1000,
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

        curriculum_term_name = getattr(
            get_command_class(self.command_term_type),
            "curriculum_term_name",
            None,
        )

        spaces: dict[str, dict[str, Any]] = {}
        for term_name, term_config in command_terms.items():

            if not isinstance(term_config, dict):
                raise TypeError(
                    f"Config of command term '{term_name}' must be a dict."
                )

            command_type = term_config.get("type")
            if not isinstance(command_type, str):
                raise ValueError(
                    f"'type' is missing for command term '{term_name}'."
                )
            if (
                getattr(
                    get_command_class(command_type),
                    "curriculum_term_name",
                    None
                ) != curriculum_term_name
            ):
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
                "num_cell": params.get("num_cell", 1),
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
        cell_intervals: list[torch.Tensor] = []
        dimension_names: list[str] = []
        term_slices: dict[str, slice] = {}
        raw_num_cells = 1
        for term_name, config in dimensions.items():

            num_cell = int(config.get("num_cell") or 1)
            if num_cell < 1:
                raise ValueError(
                    f"'num_cell' of command term '{term_name}' "
                    "must be positive."
                )

            command_dim = config.get("dim")
            if (
                not isinstance(command_dim, int)
                or isinstance(command_dim, bool)
                or command_dim <= 0
            ):
                raise ValueError(
                    f"'command_dim' of command term '{term_name}' "
                    "must be a positive integer."
                )

            raw_num_cells *= num_cell ** command_dim
            if raw_num_cells > self.max_cells:
                raise ValueError(
                    f"Curriculum space '{name}' has at least "
                    f"{raw_num_cells} raw cells, exceeding "
                    f"max_cells={self.max_cells}. Reduce 'num_cell' or "
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

            interval = (max_value - min_value) / num_cell
            cell_intervals.append(
                torch.full(
                    (command_dim,),
                    interval,
                    dtype=self.context.dtype,
                    device=self.context.device,
                )
            )
            axis = (
                torch.arange(
                    num_cell,
                    dtype=self.context.dtype,
                    device=self.context.device,
                ) * interval
                + min_value
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
        all_cell_starts = (
            axes[0].unsqueeze(-1)
            if len(axes) == 1
            else torch.cartesian_prod(*axes)
        )
        all_command_starts = {
            term_name: all_cell_starts[:, term_slice]
            for term_name, term_slice in term_slices.items()
        }

        cell_interval = torch.cat(cell_intervals)
        command_interval = {
            term_name: cell_interval[term_slice]
            for term_name, term_slice in term_slices.items()
        }

        # Keep only cells whose complete intervals satisfy every constraint.
        filtered_command_starts = constraint_set.filter(
            command_starts=all_command_starts,
            cell_interval=command_interval,
            target_names=set(dimensions),
        )
        filtered_cell_starts = torch.cat(
            [filtered_command_starts[name] for name in dimensions],
            dim=-1,
        )

        num_cells = filtered_cell_starts.shape[0]
        if num_cells == 0:
            raise ValueError(
                f"Curriculum space '{name}' has no command cells "
                "remained after applying constraints."
            )
        if num_cells > self.max_cells:
            raise ValueError(
                f"Curriculum space '{name}' has {num_cells} cells, "
                f"exceeding max_cells={self.max_cells}."
            )

        return CommandCurriculumBuffer(
            dimension_names=tuple(dimension_names),
            term_slices=term_slices,
            cell_starts=filtered_cell_starts,
            cell_interval=cell_interval,
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


    def get_command(
        self,
        space_name: str,
        dimension: str,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
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
        
        command_start = buffer.cell_starts[cell_ids, term_slice]
        command_end = command_start + buffer.cell_interval[term_slice]

        return command_start, command_end


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


# @register_curriculum
# class LrpcCommandReward(CommandRewardCurriculum):

#     def __init__(
#         self,
#         *args, **kwargs,
#     ) -> None:

#         super().__init__(
#             command_term_type="LrpcSampleOnReset",
#             *args, **kwargs
#         )


#     def update(     # update rewards to cell buffer
#         self,
#         reward: torch.Tensor
#     ) -> dict[str, torch.Tensor]:

#         info: dict[str, torch.Tensor] = {}

#         for name, buffer in self.buffers.items():

#             valid = buffer.assigned_cell_ids >= 0
#             cell_ids = buffer.assigned_cell_ids[valid]

#             buffer.reward_sum.scatter_add_(0, cell_ids, reward[valid])
#             buffer.sample_count.scatter_add_(0, cell_ids, torch.ones_like(cell_ids))

#             info[f"{name}/probabilities"] = self._get_probabilities(name)

#         return info


#     def reset(
#         self,
#         env_ids: torch.Tensor | None = None
#     ) -> None:
        
#         for buffer in self.buffers.values():

#             if env_ids is None:
#                 buffer.assigned_cell_ids.fill_(-1)
#             else:
#                 buffer.assigned_cell_ids[env_ids] = -1

#         self.resample(set(self.buffers), env_ids)


#     def resample(
#         self,
#         space_names: set[str],
#         env_ids: torch.Tensor | None = None,
#     ) -> None:
        
#         selected_count = self.num_envs if env_ids is None else env_ids.numel()

#         for space_name in space_names:

#             buffer = self.buffers[space_name]
#             sampled_ids = torch.multinomial(
#                 self._get_probabilities(space_name),    # propabilities of each cell
#                 selected_count,                         # number of cells to sample
#                 replacement=True,                       # allow duplicates
#             )

#             if env_ids is None:
#                 buffer.assigned_cell_ids.copy_(sampled_ids)
#             else:
#                 buffer.assigned_cell_ids[env_ids] = sampled_ids


#     def _get_probabilities(
#         self,
#         space_name: str
#     ) -> torch.Tensor:
        
#         buffer = self.buffers[space_name]

#         # avoid division by zero by clamping sample counts to at least 1
#         counts = buffer.sample_count.clamp_min(1).to(self.context.dtype)

#         reward_mean = buffer.reward_sum / counts
#         probabilities = torch.softmax(
#             -reward_mean / self.temperature,
#             dim=0,
#         )

#         return (
#             (1.0 - self.exploration) * probabilities
#             + self.exploration / probabilities.numel()
#         )



class BaseLpacCommandReward(CommandRewardCurriculum, ABC):

    def __init__(
        self,
        min_samples_per_cell: int = 5,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if (
            not isinstance(min_samples_per_cell, int)
            or isinstance(min_samples_per_cell, bool)
            or min_samples_per_cell <= 0
        ):
            raise ValueError(
                "'min_samples_per_cell' must be a positive integer."
            )

        self.min_samples_per_cell = min_samples_per_cell

        self.last_reward_mean = {
            name: torch.full_like(buffer.reward_sum, torch.nan)
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


    def update(
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
            episode_reward_mean = (
                self.episode_reward_sum[done]
                / self.episode_step_count[done]
            )
            buffer.reward_sum.scatter_add_(0, cell_ids, episode_reward_mean)
            buffer.sample_count.scatter_add_(0, cell_ids, torch.ones_like(cell_ids))
            buffer.assigned_cell_ids[done] = -1
            self._update_space(name)

        if env_ids is None:
            self.episode_step_count.fill_(0)
            self.episode_reward_sum.fill_(0)
        else:
            self.episode_step_count[done] = 0
            self.episode_reward_sum[done] = 0

        self.resample(set(self.buffers), env_ids)


    @abstractmethod
    def _update_space(self, space_name: str) -> None:
        raise NotImplementedError


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


@register_curriculum
class LpacCommandReward(BaseLpacCommandReward):

    def __init__(
        self,
        *args, **kwargs,
    ) -> None:

        self.min_coverage = 1.0
        super().__init__(
            command_term_type="LpacSampleOnReset",
            *args, **kwargs,
        )
        self.last_reward_mean = {
            name: torch.zeros_like(buffer.reward_sum)
            for name, buffer in self.buffers.items()
        }


    def _update_space(
        self,
        space_name: str
    ) -> None:
        
        buffer = self.buffers[space_name]
        ready = buffer.sample_count >= self.min_samples_per_cell

        if not torch.all(ready):
            return

        reward_mean = self._calculate_reward_mean(space_name)
        last_reward_mean = self.last_reward_mean[space_name]

        self.learning_progress[space_name].copy_(
            reward_mean - last_reward_mean
        )

        last_reward_mean.copy_(reward_mean)
        self.probabilities[space_name] = self._calculate_probabilities(
            space_name
        )
        buffer.sample_count.zero_()
        buffer.reward_sum.zero_()


@register_curriculum
class FastLpacCommandReward(BaseLpacCommandReward):

    def __init__(
        self,
        min_coverage: float = 0.8,
        *args, **kwargs,
    ) -> None:
        
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("'min_coverage' must be in (0, 1].")

        self.min_coverage = min_coverage
        super().__init__(
            command_term_type="FastLpacSampleOnReset",
            *args, **kwargs,
        )


    def _update_space(
        self,
        space_name: str
    ) -> None:
        
        buffer = self.buffers[space_name]
        ready = buffer.sample_count >= self.min_samples_per_cell
        if not torch.any(ready):
            return

        reward_mean = self._calculate_reward_mean(space_name)
        last_reward_mean = self.last_reward_mean[space_name]
        first_update = torch.isnan(last_reward_mean).all()

        if first_update:
            coverage = ready.count_nonzero().item() / ready.numel()
            if coverage < self.min_coverage:
                return

            baseline = reward_mean[ready].mean()
            last_reward_mean.fill_(baseline)
            last_reward_mean[ready] = reward_mean[ready]
            self.learning_progress[space_name].copy_(last_reward_mean)

        else:
            self.learning_progress[space_name][ready] = (
                reward_mean[ready] - last_reward_mean[ready]
            )
            last_reward_mean[ready] = reward_mean[ready]

        self.probabilities[space_name] = self._calculate_probabilities(
            space_name
        )
        buffer.sample_count[ready] = 0
        buffer.reward_sum[ready] = 0
