import math
import torch
from collections.abc import Sequence
from enum import Enum, auto
from numbers import Real
from typing import Any, Protocol, runtime_checkable

from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from utils.matching import resolve_metric_name
from utils.scalar import scalar_value


@runtime_checkable
class LearningRateAdjustable(Protocol):
    learning_rate: float
    optimizer: torch.optim.Optimizer


class LearningRateAdjustment(Enum):
    INCREASE = auto()
    DECREASE = auto()


class AdaptiveLearningRateCallback(BaseCallback):

    _MONITOR_ADJUSTMENT_MAP = {
        "rollout/approx_kl": (
            LearningRateAdjustment.INCREASE,
            LearningRateAdjustment.DECREASE,
        ),
        "rollout/clip_fraction": (
            LearningRateAdjustment.INCREASE,
            LearningRateAdjustment.DECREASE,
        ),
        "rollout/entropy": (
            LearningRateAdjustment.DECREASE,
            LearningRateAdjustment.INCREASE,
        ),
    }

    def __init__(
        self,
        runner: BaseRunner,
        monitor: str,
        allowed_range: Sequence[float],
        factor: float,
        min_learning_rate: float | None = None,
        max_learning_rate: float | None = None,
        *args, **kwargs,
    ) -> None:
        
        if not monitor:
            raise ValueError("Learning-rate monitor cannot be empty.")
        if len(allowed_range) != 2:
            raise ValueError("'allowed_range' must contain two values.")
        lower_bound = float(allowed_range[0])
        upper_bound = float(allowed_range[1])
        if not lower_bound < upper_bound:
            raise ValueError(
                "Learning-rate monitor lower bound must be less than "
                "the upper bound."
            )
        if not 0.0 < factor < 1.0:
            raise ValueError("'factor' must be in the open interval (0, 1).")
        if min_learning_rate is not None and min_learning_rate <= 0.0:
            raise ValueError("'min_learning_rate' must be greater than 0.")
        if max_learning_rate is not None and max_learning_rate <= 0.0:
            raise ValueError("'max_learning_rate' must be greater than 0.")
        if (
            min_learning_rate is not None
            and max_learning_rate is not None
            and min_learning_rate > max_learning_rate
        ):
            raise ValueError(
                "'min_learning_rate' cannot exceed 'max_learning_rate'."
            )

        self.runner = runner
        self.monitor = resolve_metric_name(
            info=self._MONITOR_ADJUSTMENT_MAP,
            pattern=monitor,
            owner="Learning-rate",
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.factor = factor
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self._algorithm: LearningRateAdjustable
        self._optimizer: torch.optim.Optimizer


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:

        algorithm = self.runner.algorithm
        if not isinstance(algorithm, LearningRateAdjustable):
            raise RuntimeError(
                "Algorithm must expose 'learning_rate' and a torch optimizer."
            )

        learning_rate = algorithm.learning_rate
        if (
            not isinstance(learning_rate, Real)
            or isinstance(learning_rate, bool)
            or learning_rate <= 0.0
        ):
            raise RuntimeError(
                "Algorithm must expose a positive numeric 'learning_rate'."
            )

        optimizer = algorithm.optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError("Algorithm must expose a torch optimizer.")

        self._algorithm = algorithm
        self._optimizer = optimizer
        return True


    def _on_iteration_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:
        
        if self.monitor not in info:
            raise KeyError(
                f"Training info has no learning-rate metric "
                f"{self.monitor!r}."
            )
        metric_value = scalar_value(info[self.monitor])
        if metric_value is None:
            raise TypeError(
                f"Learning-rate metric {self.monitor!r} "
                "must be a numeric scalar."
            )

        new_learning_rate = float(self._algorithm.learning_rate)
        if math.isfinite(metric_value):
            lower_adjustment, upper_adjustment = (
                self._MONITOR_ADJUSTMENT_MAP[self.monitor]
            )
            if metric_value < self.lower_bound:
                new_learning_rate = self._adjust_learning_rate(
                    new_learning_rate,
                    lower_adjustment,
                )
            elif metric_value > self.upper_bound:
                new_learning_rate = self._adjust_learning_rate(
                    new_learning_rate,
                    upper_adjustment,
                )

        if self.min_learning_rate is not None:
            new_learning_rate = max(
                new_learning_rate,
                self.min_learning_rate,
            )
        if self.max_learning_rate is not None:
            new_learning_rate = min(
                new_learning_rate,
                self.max_learning_rate,
            )

        self._algorithm.learning_rate = new_learning_rate
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = new_learning_rate

        return True


    def _adjust_learning_rate(
        self,
        learning_rate: float,
        adjustment: LearningRateAdjustment,
    ) -> float:
        if adjustment is LearningRateAdjustment.INCREASE:
            return learning_rate / self.factor
        return learning_rate * self.factor
