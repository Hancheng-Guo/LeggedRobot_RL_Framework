import torch
import operator
import statistics
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


from runners.callbacks.base import BaseCallback
from utils.matching import resolve_metric_name


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

    def __post_init__(self) -> None:
        self.values = deque(maxlen=self.window)

    @property
    def satisfied(self) -> bool:
        if len(self.values) < self.window:
            return False
        aggregated_value = self.aggregate(self.values)
        return self.compare(aggregated_value, self.threshold)


class StageCallback(BaseCallback):

    def __init__(
        self,
        condition: Sequence[Mapping[str, Any]],
        *args, **kwargs,
    ) -> None:
        self.runner = None
        self.stop_training = False
        self._conditions = self._build_conditions(
            condition
        )


    @staticmethod
    def _build_conditions(
        config: Sequence[Mapping[str, Any]],
    ) -> list[_StageCondition]:
        
        if not isinstance(config, list) or not config:
            raise ValueError("Stage condition must be a non-empty list.")

        conditions: list[_StageCondition] = []
        for detail in config:
            if not isinstance(detail, Mapping):
                raise TypeError(
                    "Each stage transition condition must be a mapping."
                )

            expected_keys = {
                "metric", "method", "operator", "threshold", "window"
            }
            if set(detail) != expected_keys:
                raise ValueError(
                    "Each stage transition condition must contain exactly: "
                    "metric, method, operator, threshold, window."
                )

            metric = detail["metric"]
            if not isinstance(metric, str) or not metric:
                raise ValueError(
                    "Stage metric must be a non-empty string."
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

            window = detail.get("window")
            if (
                not isinstance(window, int)
                or isinstance(window, bool)
                or window <= 0
            ):
                raise ValueError(
                    f"Stage window for {metric!r} must be a positive integer."
                )

            conditions.append(_StageCondition(
                metric=metric,
                aggregate=AGGREGATION_MAP[method_name],
                compare=OPERATOR_MAP[operator_name],
                threshold=float(threshold),
                window=window,
            ))
        return conditions


    def set_runner(self, runner: Any) -> None:
        self.runner = runner


    def _on_train_start(self, *args: Any, **kwargs: Any) -> bool:
        self.stop_training = False
        for condition in self._conditions:
            condition.values.clear()
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
            metric_name = resolve_metric_name(
                info=info,
                pattern=condition.metric,
                owner="Stage",
                require_match=False,
            )
            if metric_name is None:
                continue

            condition.values.append(
                self._to_scalar(info[metric_name], condition.metric)
            )

        self.stop_training = all(
            condition.satisfied
            for condition in self._conditions
        )
        return not self.stop_training


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
