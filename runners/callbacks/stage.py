import torch
import operator
import statistics
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


from runners.callbacks.base import BaseCallback


AGGREGATION_MAP: dict[
    str,
    Callable[[Iterable[float]], float],
] = {
    "mean": statistics.fmean,
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
    info_key: str
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
        condition: Mapping[str, Any],
        *args, **kwargs,
    ) -> None:
        self.runner = None
        self.stop_training = False
        self._conditions = self._build_conditions(
            dict(condition)
        )


    @staticmethod
    def _build_conditions(
        config: Mapping[str, Any],
    ) -> list[_StageCondition]:
        
        if not config:
            raise ValueError("Stage condition cannot be empty.")

        conditions: list[_StageCondition] = []
        for metric, detail in config.items():

            if not isinstance(detail, Mapping):
                raise TypeError(
                    f"Stage condition {metric!r} must be a mapping."
                )

            aggregation_name, _, reward_term = metric.partition("_")

            if not reward_term:
                raise ValueError(
                    f"Invalid stage metric: {metric!r}. Expected "
                    "'<aggregation>_<reward_term>'."
                )
            if aggregation_name not in AGGREGATION_MAP:
                valid_aggregations = ", ".join(AGGREGATION_MAP)
                raise ValueError(
                    f"Unsupported stage aggregation: "
                    f"{aggregation_name!r}. Expected one of: "
                    f"{valid_aggregations}."
                )

            info_key = (
                "reward"
                if reward_term == "reward"
                else f"reward/{reward_term}"
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
                info_key=info_key,
                aggregate=AGGREGATION_MAP[aggregation_name],
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
        
        if self.stop_training:
            return False

        for condition in self._conditions:
            if condition.info_key not in info:
                raise KeyError(
                    f"Training info is missing {condition.info_key!r} "
                    f"required by stage metric {condition.metric!r}."
                )
            condition.values.append(
                self._to_scalar(info[condition.info_key], condition.metric)
            )

        self.stop_training = all(
            condition.satisfied
            for condition in self._conditions
        )
        return not self.stop_training


    @staticmethod
    def _to_scalar(
        value: Any,
        metric: str
    ) -> float:
        
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Stage metric {metric!r} must be scalar, got tensor "
                    f"shape {tuple(value.shape)}."
                )
            return float(value.detach().item())
        if isinstance(value, Real) and not isinstance(value, bool):
            return float(value)
        raise TypeError(f"Stage metric {metric!r} must be numeric.")
