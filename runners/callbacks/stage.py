"""Stage configuration, condition state, and callback sampling."""

import torch
import math
import operator
import statistics
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from glob import has_magic
from numbers import Real
from typing import Any

from .base import BaseCallback


ConditionValues = Mapping[str, Sequence[float]]


AGGREGATION_MAP: dict[
    str,
    Callable[[Iterable[float]], float],
] = {
    "mean": statistics.fmean,
    "max": max,
    "min": min,
}


OPERATOR_MAP: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(slots=True)
class _StageCondition:
    metric: str
    aggregate: Callable[[Iterable[float]], float]
    compare: Callable[[float, float], bool]
    threshold: float
    window: int
    values: deque[float] = field(init=False)
    last_revision: int | None = None
    # Curriculum-only controls; leave disabled for reward/rollout metrics.
    sample_on_update: bool = False
    require_all_cells_valid: bool = False


    def __post_init__(self) -> None:
        self.values = deque(maxlen=self.window)


    @property
    def satisfied(self) -> bool:
        return (
            len(self.values) == self.window
            and self.compare(self.aggregate(self.values), self.threshold)
        )


    @property
    def uses_curriculum_controls(self) -> bool:
        return self.sample_on_update or self.require_all_cells_valid


    def reset(self) -> None:
        self.values.clear()
        self.last_revision = None


    def curriculum_control_names(self) -> tuple[str | None, str | None]:
        prefix = self.metric.rsplit("/", 1)[0]
        revision = f"{prefix}/statistics_revision" if self.sample_on_update else None
        validity = f"{prefix}/all_cells_valid" if self.require_all_cells_valid else None
        return revision, validity


    def get_values(self) -> ConditionValues:
        return {self.metric: list(self.values)}


    def restore_values(self, saved: ConditionValues) -> None:
        if not isinstance(saved, Mapping):
            raise TypeError(
                "Checkpoint condition values must map metric names to sample sequences."
            )
        self.reset()
        for name, samples in saved.items():
            if name != self.metric:
                raise ValueError(
                    f"Checkpoint metric {name!r} does not match {self.metric!r}."
                )
            self.values.extend(float(value) for value in samples)


def build_conditions(config: Sequence[Mapping[str, Any]]) -> list[_StageCondition]:
    if not isinstance(config, list) or not config:
        raise ValueError("Stage condition must be a non-empty list.")
    return [_build_condition(detail) for detail in config]


def _build_condition(detail: Mapping[str, Any]) -> _StageCondition:
    if not isinstance(detail, Mapping):
        raise TypeError(
            "Each stage transition condition must be a mapping."
        )

    expected_keys = {
        "metric", "method", "operator", "threshold", "window"
    }
    optional_keys = {"sample_on_update", "require_all_cells_valid"}
    if not expected_keys <= set(detail) or set(detail) - (expected_keys | optional_keys):
        raise ValueError(
            "Each stage transition condition must contain: "
            "metric, method, operator, threshold, window; "
            "optional fields: sample_on_update, require_all_cells_valid."
        )

    metric = detail["metric"]
    if not isinstance(metric, str) or not metric:
        raise ValueError(
            "Stage metric must be a non-empty string."
        )
    if has_magic(metric):
        raise ValueError("Stage metric must be an exact metric name; wildcards are not supported.")

    for key in optional_keys:
        if key in detail and not isinstance(detail[key], bool):
            raise TypeError(f"Stage {key} must be a bool.")

    if any(detail.get(key, False) for key in optional_keys):
        parts = metric.split("/")
        if (
            len(parts) != 4
            or parts[0] != "curriculum"
            or not all(parts[1:])
        ):
            raise ValueError(
                "Stage curriculum options require "
                "'curriculum/<term>/<group>/<metric>' with explicit "
                "term and group names."
            )

    method_name = detail["method"]
    if (
        not isinstance(method_name, str)
        or method_name not in AGGREGATION_MAP
    ):
        valid_aggregations = ", ".join(AGGREGATION_MAP)
        raise ValueError(
            f"Unsupported stage method: "
            f"{method_name!r}. Expected one of: "
            f"{valid_aggregations}."
        )

    operator_name = detail.get("operator")
    if operator_name not in OPERATOR_MAP:
        valid_operators = ", ".join(OPERATOR_MAP)
        raise ValueError(
            f"Unsupported stage operator: {operator_name!r}. "
            f"Expected one of: {valid_operators}."
        )

    threshold = detail.get("threshold")
    if (
        not isinstance(threshold, Real)
        or isinstance(threshold, bool)
    ):
        raise TypeError(
            f"Stage threshold for {metric!r} must be numeric."
        )
    if not math.isfinite(float(threshold)):
        raise ValueError(f"Stage threshold for {metric!r} must be finite.")

    window = detail.get("window")
    if (
        not isinstance(window, int)
        or isinstance(window, bool)
        or window <= 0
    ):
        raise ValueError(
            f"Stage window for {metric!r} must be a positive integer."
        )

    return _StageCondition(
        metric=metric,
        aggregate=AGGREGATION_MAP[method_name],
        compare=OPERATOR_MAP[operator_name],
        threshold=float(threshold),
        window=window,
        sample_on_update=detail.get("sample_on_update", False),
        require_all_cells_valid=detail.get("require_all_cells_valid", False),
    )


class StageCallback(BaseCallback):

    def __init__(
        self,
        condition: Sequence[Mapping[str, Any]],
        *args, **kwargs,
    ) -> None:
        self.runner = None
        self.stop_training = False
        self._conditions = build_conditions(condition)


    def set_runner(self, runner: Any) -> None:
        self.runner = runner


    def _on_train_start(self, *args, **kwargs) -> bool:
        self.stop_training = False
        for condition in self._conditions:
            condition.reset()
        self._validate_curriculum_metrics()
        return True


    def _validate_curriculum_metrics(self) -> None:

        conditions = [
            condition for condition in self._conditions
            if condition.metric.startswith("curriculum/")
        ]
        if not conditions:
            return

        environment = getattr(self.runner, "environment", None)
        task = getattr(environment, "task", None)
        curriculum_manager = getattr(task, "curriculum_manager", None)
        if curriculum_manager is None:
            raise ValueError(
                "Stage curriculum metrics require a task with a curriculum manager."
            )

        available = curriculum_manager.get_metric_names()
        for condition in conditions:
            if condition.metric not in available:
                raise KeyError(
                    f"Stage curriculum metric {condition.metric!r} is not a supported metric; "
                    f"available: {', '.join(sorted(available))}."
                )
            for control in condition.curriculum_control_names():
                if control is not None and control not in available:
                    raise KeyError(
                        f"Stage condition {condition.metric!r} requires unsupported metric {control!r}."
                    )


    def _on_step_end(
        self,
        info: Mapping[str, Any],
        *args, **kwargs,
    ) -> bool:
        return self._update_conditions(info)


    def _on_iteration_end(
        self,
        info: Mapping[str, Any],
        *args, **kwargs,
    ) -> bool:
        return self._update_conditions(info)


    def _update_conditions(
        self,
        info: Mapping[str, Any]
    ) -> bool:

        if self.stop_training:
            return False

        for condition in self._conditions:
            self._sample_condition(condition, info)

        self.stop_training = all(
            condition.satisfied
            for condition in self._conditions
        )
        return not self.stop_training


    def _sample_condition(
        self,
        condition: _StageCondition,
        info: Mapping[str, Any],
    ) -> None:

        metric_name = condition.metric
        revision_name, validity_name = condition.curriculum_control_names()
        if not any(
            key is not None and key in info
            for key in (metric_name, revision_name, validity_name)
        ):
            return

        is_new_revision = self._check_revision(condition, info, revision_name)
        if not self._check_validity(condition, info, validity_name):
            return

        value = self._read_metric(condition, info)
        if value is not None and is_new_revision:
            condition.values.append(value)


    def _check_revision(
        self, condition: _StageCondition,
        info: Mapping[str, Any],
        revision_name: str | None,
    ) -> bool:

        if revision_name is None:
            return True
        if revision_name not in info:
            raise KeyError(
                f"Stage condition {condition.metric!r} requires control metric "
                f"{revision_name!r} when sample_on_update is enabled."
            )

        value = self._control_scalar(info[revision_name], revision_name)
        if (
            not math.isfinite(value)
            or value < 0 or
            not value.is_integer()
        ):
            raise ValueError(
                f"Stage revision metric {revision_name!r} must be a finite non-negative integer, "
                f"got {value!r}."
            )

        revision = int(value)
        if revision == condition.last_revision:
            return False

        if (
            condition.last_revision is not None
            and revision < condition.last_revision
        ):
            condition.values.clear()
        condition.last_revision = revision
        return True


    def _check_validity(
        self, condition: _StageCondition,
        info: Mapping[str, Any],
        validity_name: str | None,
    ) -> bool:

        if validity_name is None:
            return True
        if validity_name not in info:
            raise KeyError(
                f"Stage condition {condition.metric!r} requires control metric "
                f"{validity_name!r} when require_all_cells_valid is enabled."
            )
        if self._control_scalar(info[validity_name], validity_name) != 1.0:
            condition.values.clear()
            return False
        return True


    def _read_metric(
        self,
        condition: _StageCondition,
        info: Mapping[str, Any],
    ) -> float | None:

        metric_name = condition.metric
        if metric_name not in info:
            if condition.require_all_cells_valid:
                raise KeyError(
                    f"Stage metric {metric_name!r} is missing despite all_cells_valid being 1."
                )
            condition.values.clear()
            return None
        value = self._to_scalar(info[metric_name], condition.metric)
        if not math.isfinite(value):
            condition.values.clear()
            return None
        return value


    def _control_scalar(
        self,
        value: Any,
        metric: str
    ) -> float:

        if isinstance(value, torch.Tensor) and value.numel() != 1:
            raise ValueError(f"Stage control metric {metric!r} must be scalar.")
        if isinstance(value, bool):
            return float(value)
        return self._to_scalar(value, metric)


    def _to_scalar(
        self,
        value: Any,
        metric: str
    ) -> float:

        if isinstance(value, torch.Tensor):
            value_count = value.numel()
            if value_count == 0:
                raise ValueError(
                    f"Stage metric {metric!r} must not be empty."
                )
            if value_count == 1:
                return float(value.detach().item())

            if self.runner is None or not hasattr(self.runner, "environment"):
                raise RuntimeError(
                    "StageCallback must be attached to a runner before "
                    "processing a vector metric."
                )

            num_envs = self.runner.environment.num_envs
            if value_count != num_envs:
                raise ValueError(
                    f"Stage metric {metric!r} must contain either one value "
                    f"or num_envs ({num_envs}) values, got {value_count}."
                )
            return float(value.detach().float().mean().item())

        if isinstance(value, Real) and not isinstance(value, bool):
            return float(value)

        raise TypeError(f"Stage metric {metric!r} must be numeric.")


    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "stop_training": self.stop_training,
            "condition_values": [
                condition.get_values()
                for condition in self._conditions
            ],
            "condition_revisions": [
                condition.last_revision
                for condition in self._conditions
            ],
        }


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        environment_restored: bool = False,
    ) -> None:

        saved: Sequence[ConditionValues] = state.get(
            "condition_values", ()
        )
        if len(saved) != len(self._conditions):
            raise ValueError(
                "Checkpoint stage-condition count does not match configuration."
            )

        revisions = state.get("condition_revisions", [None] * len(self._conditions))
        if not isinstance(revisions, Sequence) or len(revisions) != len(self._conditions):
            raise ValueError("Checkpoint stage-condition revision count does not match configuration.")

        discarded_history = False
        for condition, values, revision in zip(self._conditions, saved, revisions):
            if condition.uses_curriculum_controls and not environment_restored:
                condition.reset()
                discarded_history = True
                continue
            self._restore_condition(condition, values, revision)
        self.stop_training = bool(state.get("stop_training", False)) and not discarded_history


    def _restore_condition(
        self,
        condition: _StageCondition,
        values: ConditionValues,
        revision: Any,
    ) -> None:
        if revision is not None and (
            not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
        ):
            raise ValueError("Checkpoint stage revision must be a non-negative integer or None.")
        condition.restore_values(values)
        if not all(math.isfinite(value) for value in condition.values):
            raise ValueError("Checkpoint stage samples must be finite.")
        if condition.sample_on_update and condition.values and revision is None:
            raise ValueError("Checkpoint stage samples require their last statistics revision.")
        condition.last_revision = revision if condition.sample_on_update else None
