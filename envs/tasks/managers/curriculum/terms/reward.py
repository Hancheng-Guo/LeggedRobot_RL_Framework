import math
import torch
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from numbers import Real
from collections.abc import Mapping
from copy import deepcopy

from .base import BaseCurriculumTerm
from .registry import register_curriculum
from ..types import CurriculumSpace


@dataclass
class CurriculumBuffer:
    dimension_names: tuple[str, ...]
    term_slices: dict[str, slice]
    cell_starts: torch.Tensor
    cell_interval: torch.Tensor
    reward_sum: torch.Tensor
    sample_count: torch.Tensor
    assigned_cell_ids: torch.Tensor


@dataclass
class CellRewardStatistics:
    reward_sum: torch.Tensor
    sample_count: torch.Tensor
    reward_mean: torch.Tensor
    valid: torch.Tensor
    revision: torch.Tensor


class RewardCurriculum(BaseCurriculumTerm, ABC):

    manager_config_names = ("reward_manager_config",)
    requires_sampling_space = True

    def __init__(
        self,
        sampling_space: CurriculumSpace,
        reward_manager_config: dict[str, Any],
        reward_sources: list[str | dict[str, float]],
        temperature: float = 0.01,
        exploration: float = 0.1,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if temperature <= 0.0:
            raise ValueError("'temperature' must be greater than 0.")
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("'exploration' must be between 0 and 1.")

        self.temperature = temperature
        self.exploration = exploration
        self.space = sampling_space
        self.reward_sources = self._parse_reward_sources(
            reward_sources,
            reward_manager_config,
        )
        self.buffer = self._build_buffer()
        self._source_config = deepcopy(sampling_space.source_config)
        self._reward_config = deepcopy(reward_manager_config)


    def _parse_reward_sources(
        self,
        reward_sources: list[str | dict[str, float]],
        reward_manager_config: dict[str, Any],
    ) -> dict[str, float | None]:
        """Resolve weighted rewards and explicit raw-reward coefficients."""

        owner = f"Curriculum {type(self).__name__!r}"
        label = f"{owner} reward_sources"
        if not isinstance(reward_sources, list) or not reward_sources:
            raise ValueError(f"{label} must be a non-empty list.")

        if not isinstance(reward_manager_config, dict):
            raise TypeError(f"{owner} 'reward_manager_config' must be a dict.")
        reward_terms = reward_manager_config.get("terms")
        if not isinstance(reward_terms, dict):
            raise TypeError(f"{owner} reward manager must define a 'terms' dict.")

        coefficients: dict[str, float | None] = {}
        for entry in reward_sources:
            override = isinstance(entry, dict)
            if isinstance(entry, str):
                name, coefficient = entry, None
            elif override and len(entry) == 1:
                name, coefficient = next(iter(entry.items()))
            else:
                raise ValueError(
                    f"{label} entries must be strings or single-entry dicts."
                )
            if not isinstance(name, str) or not name:
                raise ValueError(f"{label} reward names must be non-empty strings.")

            if name != "reward" and (
                not name.startswith("reward/")
                or name.removeprefix("reward/") not in reward_terms
            ):
                raise ValueError(f"{label} references unknown reward source {name!r}.")
            if name in coefficients:
                raise ValueError(f"{label} has duplicate reward source {name!r}.")
            if override:
                if name == "reward":
                    raise ValueError(f"{label} 'reward' does not support an override coefficient.")
                if not isinstance(coefficient, Real) or isinstance(coefficient, bool):
                    raise TypeError(f"{label} reward {name!r} coefficient must be numeric.")
                coefficient = float(coefficient)
                if not math.isfinite(coefficient):
                    raise ValueError(f"{label} reward {name!r} coefficient must be finite.")
            coefficients[name] = coefficient

        if all(coefficient == 0.0 for coefficient in coefficients.values()):
            raise ValueError(f"{label} coefficients must not all be zero.")

        return coefficients


    def _build_buffer(self) -> CurriculumBuffer:

        space = self.space
        num_cells = space.cell_starts.shape[0]

        return CurriculumBuffer(
            dimension_names=space.dimension_names,
            term_slices=dict(space.term_slices),
            cell_starts=space.cell_starts.to(device=self.context.device, dtype=self.context.dtype).clone(),
            cell_interval=space.cell_interval.to(device=self.context.device, dtype=self.context.dtype).clone(),
            reward_sum=torch.zeros(num_cells, device=self.context.device, dtype=self.context.dtype),
            sample_count=torch.zeros(num_cells, device=self.context.device, dtype=torch.long),
            assigned_cell_ids=torch.full((self.num_envs,), -1, device=self.context.device, dtype=torch.long),
        )


    def get_command(
        self,
        dimension: str,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        buffer = self.buffer
        if dimension not in buffer.term_slices:
            raise ValueError(f"Unknown dimension '{dimension}' in curriculum space.")
        term_slice = buffer.term_slices[dimension]
        cell_ids = buffer.assigned_cell_ids
        if env_ids is not None:
            cell_ids = cell_ids[env_ids]
        if torch.any(cell_ids < 0):
            raise RuntimeError("Curriculum space has not been sampled.")
        lower = buffer.cell_starts[cell_ids, term_slice]
        upper = lower + buffer.cell_interval[term_slice]
        return lower, upper


class BaseLpacReward(RewardCurriculum, ABC):

    min_coverage: float

    def __init__(
        self,
        min_samples_per_cell: int = 5,
        normalize_by_episode_steps: bool = False,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        if (
            not isinstance(min_samples_per_cell, int)
            or isinstance(min_samples_per_cell, bool)
            or min_samples_per_cell <= 0
        ):
            raise ValueError("'min_samples_per_cell' must be a positive integer.")
        if not isinstance(normalize_by_episode_steps, bool):
            raise TypeError("'normalize_by_episode_steps' must be a bool.")

        self.min_samples_per_cell = min_samples_per_cell
        self.normalize_by_episode_steps = normalize_by_episode_steps
        buffer = self.buffer
        # Real cell evaluations are independent of algorithm-window resets and
        # FastLPAC's estimated baselines.
        self.cell_statistics = CellRewardStatistics(
            reward_sum=torch.zeros_like(buffer.reward_sum),
            sample_count=torch.zeros_like(buffer.sample_count),
            reward_mean=torch.full_like(buffer.reward_sum, torch.nan),
            valid=torch.zeros_like(buffer.sample_count, dtype=torch.bool),
            revision=torch.zeros((), dtype=torch.long, device=self.context.device),
        )
        self._cell_metrics = self._summarize_cell_statistics()
        self.last_reward_mean = torch.full_like(buffer.reward_sum, torch.nan)
        self.learning_progress = torch.zeros_like(buffer.reward_sum)
        self.probabilities = self._calculate_probabilities()
        self.episode_reward_sum = torch.zeros(
            self.num_envs, dtype=self.context.dtype, device=self.context.device,
        )
        self.episode_step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.context.device,
        )


    def update(
        self,
        *,
        raw_term_rewards: dict[str, torch.Tensor],
        term_rewards: dict[str, torch.Tensor],
        total_reward: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        if not self.training:
            return {}

        # Resolve every input before changing accumulators. Explicit
        # coefficients apply to raw rewards; no input has dt scaling.
        reward = torch.zeros_like(self.episode_reward_sum)
        for source, coefficient in self.reward_sources.items():
            if source == "reward":
                reward += total_reward
            else:
                term_name = source.removeprefix("reward/")
                if coefficient is None:
                    reward += term_rewards[term_name]
                else:
                    reward += raw_term_rewards[term_name] * coefficient
        self.episode_reward_sum += reward
        self.episode_step_count += 1

        return {}


    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        selected = (
            torch.arange(self.num_envs, dtype=torch.long, device=self.context.device)
            if env_ids is None else env_ids
        )
        buffer = self.buffer
        if self.training:
            collected = self.episode_step_count > 0
            valid = buffer.assigned_cell_ids >= 0
            done = selected[(collected & valid)[selected]]
            cell_ids = buffer.assigned_cell_ids[done]
            episode_reward = self.episode_reward_sum[done]
            if self.normalize_by_episode_steps:
                episode_reward = episode_reward / self.episode_step_count[done]
            buffer.reward_sum.scatter_add_(0, cell_ids, episode_reward)
            buffer.sample_count.scatter_add_(0, cell_ids, torch.ones_like(cell_ids))
            self._record_cell_statistics(cell_ids, episode_reward)
            self._update_space()

        buffer.assigned_cell_ids[selected] = -1
        self.episode_step_count[selected] = 0
        self.episode_reward_sum[selected] = 0


    def _record_cell_statistics(
        self,
        cell_ids: torch.Tensor,
        episode_reward: torch.Tensor,
    ) -> None:

        stats = self.cell_statistics
        stats.reward_sum.scatter_add_(0, cell_ids, episode_reward)
        stats.sample_count.scatter_add_(0, cell_ids, torch.ones_like(cell_ids))
        ready = stats.sample_count >= self.min_samples_per_cell
        if not torch.any(ready):
            return

        means = stats.reward_sum[ready] / stats.sample_count[ready]
        stats.reward_mean[ready] = means
        stats.valid[ready] = torch.isfinite(means)
        stats.reward_sum[ready] = 0
        stats.sample_count[ready] = 0
        stats.revision.add_(1)
        self._cell_metrics = self._summarize_cell_statistics()


    def _summarize_cell_statistics(self) -> dict[str, torch.Tensor]:

        stats = self.cell_statistics
        valid_count = stats.valid.count_nonzero()
        metrics = {
            "coverage": valid_count.to(self.context.dtype) / stats.valid.numel(),
            "valid_cell_count": valid_count,
            "num_cells": stats.sample_count.new_tensor(stats.valid.numel()),
            "all_cells_valid": stats.valid.all().to(self.context.dtype),
            "statistics_revision": stats.revision.clone(),
        }
        values = stats.reward_mean[stats.valid]
        if values.numel():
            metrics.update({
                "reward_mean": values.mean(),
                "reward_min": values.min(),
                "reward_max": values.max(),
                "reward_std": values.std(unbiased=False),
            })
        return metrics


    def collect_metrics(self) -> dict[str, torch.Tensor]:
        # Return snapshots: later updates must not rewrite old info.
        return {
            name: value.clone()
            for name, value in self._cell_metrics.items()
        } | {
            "learning_progress": self.learning_progress.clone(),
            "probabilities": self.probabilities.clone(),
        }


    def get_metric_names(self) -> set[str]:
        return {
            "coverage", "valid_cell_count", "num_cells", "all_cells_valid",
            "statistics_revision", "reward_mean", "reward_min", "reward_max",
            "reward_std", "learning_progress", "probabilities",
        }


    def _checkpoint_config(self) -> dict[str, Any]:
        return {
            **self._source_config,
            "reward": self._reward_config,
            "reward_sources": self.reward_sources,
            "temperature": self.temperature,
            "exploration": self.exploration,
            "min_samples_per_cell": self.min_samples_per_cell,
            "normalize_by_episode_steps": self.normalize_by_episode_steps,
            "min_coverage": self.min_coverage,
        }


    def _checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        buffer = self.buffer
        stats = self.cell_statistics
        return {
            "reward_sum": buffer.reward_sum,
            "sample_count": buffer.sample_count,
            "last_reward_mean": self.last_reward_mean,
            "learning_progress": self.learning_progress,
            "probabilities": self.probabilities,
            "statistics_reward_sum": stats.reward_sum,
            "statistics_sample_count": stats.sample_count,
            "cell_reward_mean": stats.reward_mean,
            "cell_valid": stats.valid,
            "statistics_revision": stats.revision,
        }


    def checkpoint_state_dict(self) -> dict[str, Any]:
        buffer = self.buffer
        return super().checkpoint_state_dict() | {
            "config": deepcopy(self._checkpoint_config()),
            "space": {
                "dimension_names": buffer.dimension_names,
                "term_slices": {
                    key: (value.start, value.stop)
                    for key, value in buffer.term_slices.items()
                },
                "cell_starts": buffer.cell_starts.detach().cpu().clone(),
                "cell_interval": buffer.cell_interval.detach().cpu().clone(),
                "tensors": {
                    key: value.detach().cpu().clone()
                    for key, value in self._checkpoint_tensors().items()
                },
            },
        }


    def validate_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:

        super().validate_checkpoint_state_dict(state)

        if state.get("config") != self._checkpoint_config():
            raise ValueError(
                "Curriculum checkpoint configuration does not match current configuration."
            )
        saved = state.get("space")
        buffer = self.buffer
        slices = {key: (value.start, value.stop) for key, value in buffer.term_slices.items()}
        if (
            not isinstance(saved, Mapping)
            or saved.get("dimension_names") != buffer.dimension_names
            or saved.get("term_slices") != slices
        ):
            raise ValueError("Checkpoint cell dimensions do not match curriculum space.")
        for key, current in (
            ("cell_starts", buffer.cell_starts), ("cell_interval", buffer.cell_interval)
        ):
            value = saved.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != current.shape
                or not torch.allclose(
                    value.to(device="cpu", dtype=torch.float64),
                    current.detach().to(device="cpu", dtype=torch.float64)
                )
            ):
                raise ValueError(f"Checkpoint {key} does not match curriculum space.")
        self._validate_checkpoint_tensors(saved.get("tensors"))


    def _validate_checkpoint_tensors(
        self,
        saved: Any,
    ) -> None:

        current = self._checkpoint_tensors()
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            raise ValueError("Checkpoint tensor fields do not match curriculum space.")
        for key, target in current.items():
            value = saved[key]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError(f"Invalid checkpoint tensor shape for {key!r}.")
            valid_dtype = (
                value.is_floating_point()
                if target.is_floating_point()
                else value.dtype == target.dtype
            )
            if not valid_dtype:
                raise ValueError(f"Invalid checkpoint tensor dtype for {key!r}.")
            if (
                key in {"sample_count", "statistics_sample_count", "statistics_revision"}
                and torch.any(value < 0)
            ):
                raise ValueError(f"Checkpoint {key!r} must be non-negative.")
        probabilities = saved["probabilities"]
        if (
            not torch.isfinite(probabilities).all()
            or torch.any(probabilities < 0)
            or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.0))
        ):
            raise ValueError(
                "Checkpoint probabilities are invalid for curriculum space."
            )
        if not torch.isfinite(saved["cell_reward_mean"][saved["cell_valid"]]).all():
            raise ValueError(
                "Checkpoint valid cell rewards must be finite for curriculum space."
            )


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:

        self.validate_checkpoint_state_dict(state)
        with torch.no_grad():
            saved = state["space"]["tensors"]
            for key, target in self._checkpoint_tensors().items():
                target.copy_(saved[key].to(
                    device=target.device,
                    dtype=target.dtype
                ))
            self.buffer.assigned_cell_ids.fill_(-1)
            self.episode_reward_sum.zero_()
            self.episode_step_count.zero_()
            self._cell_metrics = self._summarize_cell_statistics()


    @abstractmethod
    def _update_space(self) -> None:
        raise NotImplementedError


    def resample(
        self,
        env_ids: torch.Tensor | None = None
    ) -> None:

        if env_ids is None:
            selected_count = self.num_envs
            active_steps = self.episode_step_count
        else:
            selected_count = env_ids.numel()
            active_steps = self.episode_step_count[env_ids]
        if torch.any(active_steps > 0):
            raise RuntimeError(
                "Settle active curriculum episodes before resampling."
            )
        if selected_count == 0:
            return
        sampled_ids = torch.multinomial(
            self._get_probabilities(), selected_count, replacement=True,
        )
        if env_ids is None:
            self.buffer.assigned_cell_ids.copy_(sampled_ids)
        else:
            self.buffer.assigned_cell_ids[env_ids] = sampled_ids


    def _get_probabilities(self) -> torch.Tensor:
        return self.probabilities


    def _calculate_reward_mean(self) -> torch.Tensor:

        buffer = self.buffer
        counts = buffer.sample_count.clamp_min(1).to(self.context.dtype)
        reward_mean = buffer.reward_sum / counts

        return reward_mean


    def _calculate_probabilities(self) -> torch.Tensor:

        probabilities = torch.softmax(
            self.learning_progress / self.temperature,
            dim=0,
        )

        return (
            (1.0 - self.exploration) * probabilities
            + self.exploration / probabilities.numel()
        )


@register_curriculum
class LpacReward(BaseLpacReward):

    def __init__(
        self,
        *args, **kwargs,
    ) -> None:

        self.min_coverage = 1.0
        super().__init__(*args, **kwargs)
        self.last_reward_mean = torch.zeros_like(self.buffer.reward_sum)


    def _update_space(self) -> None:

        buffer = self.buffer
        ready = buffer.sample_count >= self.min_samples_per_cell

        if not torch.all(ready):
            return

        reward_mean = self._calculate_reward_mean()
        last_reward_mean = self.last_reward_mean

        self.learning_progress.copy_(
            reward_mean - last_reward_mean
        )

        last_reward_mean.copy_(reward_mean)
        self.probabilities = self._calculate_probabilities()
        buffer.sample_count.zero_()
        buffer.reward_sum.zero_()


@register_curriculum
class FastLpacReward(BaseLpacReward):

    def __init__(
        self,
        min_coverage: float = 0.8,
        *args, **kwargs,
    ) -> None:

        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("'min_coverage' must be in (0, 1].")

        self.min_coverage = min_coverage
        super().__init__(*args, **kwargs)


    def _update_space(self) -> None:

        buffer = self.buffer
        ready = buffer.sample_count >= self.min_samples_per_cell
        if not torch.any(ready):
            return

        reward_mean = self._calculate_reward_mean()
        last_reward_mean = self.last_reward_mean
        first_update = torch.isnan(last_reward_mean).all()

        if first_update:
            coverage = ready.count_nonzero().item() / ready.numel()
            if coverage < self.min_coverage:
                return

            baseline = reward_mean[ready].mean()
            last_reward_mean.fill_(baseline)
            last_reward_mean[ready] = reward_mean[ready]
            self.learning_progress.copy_(last_reward_mean)

        else:
            self.learning_progress[ready] = (
                reward_mean[ready] - last_reward_mean[ready]
            )
            last_reward_mean[ready] = reward_mean[ready]

        self.probabilities = self._calculate_probabilities()
        buffer.sample_count[ready] = 0
        buffer.reward_sum[ready] = 0
