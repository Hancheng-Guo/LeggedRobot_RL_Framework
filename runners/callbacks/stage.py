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
    revision_metric: str | None = None
    validity_metric: str | None = None


    def __post_init__(self) -> None:
        self.values = deque(maxlen=self.window)


    @property
    def satisfied(self) -> bool:
        return (
            len(self.values) == self.window
            and self.compare(self.aggregate(self.values), self.threshold)
        )


    @property
    def uses_control_metrics(self) -> bool:
        return self.revision_metric is not None or self.validity_metric is not None


    def reset(self) -> None:
        self.values.clear()
        self.last_revision = None


    def get_metric_config(self) -> dict[str, str | None]:
        return {
            "metric": self.metric,
            "revision_metric": self.revision_metric,
            "validity_metric": self.validity_metric,
        }


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
    optional_keys = {"revision_metric", "validity_metric"}
    if not expected_keys <= set(detail) or set(detail) - (expected_keys | optional_keys):
        raise ValueError(
            "Each stage transition condition must contain: "
            "metric, method, operator, threshold, window; "
            "optional fields: revision_metric, validity_metric."
        )

    metric = detail["metric"]
    if not isinstance(metric, str) or not metric:
        raise ValueError(
            "Stage metric must be a non-empty string."
        )
    if has_magic(metric):
        raise ValueError(
            "Stage metric must be an exact metric name; wildcards are not supported."
        )

    for key in optional_keys:
        value = detail.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise TypeError(
                f"Stage {key} must be a non-empty string or None."
            )
        if has_magic(value):
            raise ValueError(
                f"Stage {key} must be an exact metric name; wildcards are not supported."
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
        revision_metric=detail.get("revision_metric"),
        validity_metric=detail.get("validity_metric"),
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
        return True


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

        metric_names = (
            condition.metric,
            condition.revision_metric,
            condition.validity_metric,
        )
        if not any(name is not None and name in info for name in metric_names):
            return

        # Validate both controls before changing the window or revision.
        revision = self._read_revision(condition, info)
        valid = self._read_validity(condition, info)
        if revision is None:
            should_sample = True
        else:
            previous_revision = condition.last_revision
            should_sample = revision != previous_revision

            if previous_revision is not None and revision < previous_revision:
                condition.values.clear()
            condition.last_revision = revision

        if not valid:
            condition.values.clear()
            return

        value = self._read_metric(condition, info)
        if value is not None and should_sample:
            condition.values.append(value)


    def _read_revision(
        self,
        condition: _StageCondition,
        info: Mapping[str, Any],
    ) -> int | None:

        name = condition.revision_metric
        if name is None:
            return None
        if name not in info:
            raise KeyError(
                f"Stage condition {condition.metric!r} requires revision metric {name!r}."
            )
        value = self._control_scalar(info[name], name)
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(
                f"Stage revision metric {name!r} must be a finite non-negative integer, "
                f"got {value!r}."
            )
        return int(value)


    def _read_validity(
        self,
        condition: _StageCondition,
        info: Mapping[str, Any],
    ) -> bool:

        name = condition.validity_metric
        if name is None:
            return True
        if name not in info:
            raise KeyError(
                f"Stage condition {condition.metric!r} requires validity metric {name!r}."
            )
        value = self._control_scalar(info[name], name)
        if value not in (0.0, 1.0):
            raise ValueError(
                f"Stage validity metric {name!r} must be a boolean or 0/1, got {value!r}."
            )
        return value == 1.0


    def _read_metric(
        self,
        condition: _StageCondition,
        info: Mapping[str, Any],
    ) -> float | None:

        metric_name = condition.metric
        if metric_name not in info:
            if condition.validity_metric is not None:
                raise KeyError(
                    f"Stage metric {metric_name!r} is missing despite validity metric "
                    f"{condition.validity_metric!r} being 1."
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
            "condition_metrics": [
                condition.get_metric_config()
                for condition in self._conditions
            ],
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

        if not isinstance(state, Mapping):
            raise TypeError("Stage checkpoint must be a mapping.")

        expected_metrics = [
            condition.get_metric_config()
            for condition in self._conditions
        ]
        if state.get("condition_metrics") != expected_metrics:
            raise ValueError(
                "Checkpoint stage metric references do not match configuration."
            )

        saved: Sequence[ConditionValues] = state.get(
            "condition_values", ()
        )
        if len(saved) != len(self._conditions):
            raise ValueError(
                "Checkpoint stage-condition count does not match configuration."
            )

        revisions = state.get("condition_revisions")
        if not isinstance(revisions, Sequence) or len(revisions) != len(self._conditions):
            raise ValueError(
                "Checkpoint stage-condition revision count does not match configuration."
            )

        discarded_history = False
        for condition, values, revision in zip(self._conditions, saved, revisions):
            if condition.uses_control_metrics and not environment_restored:
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
            raise ValueError(
                "Checkpoint stage revision must be a non-negative integer or None."
            )
        condition.restore_values(values)
        if not all(math.isfinite(value) for value in condition.values):
            raise ValueError("Checkpoint stage samples must be finite.")
        if condition.revision_metric is not None and condition.values and revision is None:
            raise ValueError("Checkpoint stage samples require their last revision.")
        condition.last_revision = revision if condition.revision_metric is not None else None
