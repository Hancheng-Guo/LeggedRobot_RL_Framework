import math
from collections.abc import Mapping
from typing import Any, Literal

from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from utils.matching import resolve_metric_name
from utils.scalar import scalar_value


class EarlystoppingCallback(BaseCallback):

    def __init__(
        self,
        runner: BaseRunner,
        monitor: str,
        max_no_improve_iters: int,
        mode: Literal["min", "max"],
        min_delta: float,
        warmup_iters: int = 0,
        *args, **kwargs,
    ) -> None:
        
        if not monitor:
            raise ValueError("Early-stopping 'monitor' cannot be empty.")
        if mode not in ("min", "max"):
            raise ValueError("Early-stopping 'mode' must be 'min' or 'max'.")
        if max_no_improve_iters <= 0:
            raise ValueError(
                "'max_no_improve_iters' must be greater than 0."
            )
        if min_delta < 0.0:
            raise ValueError("'min_delta' must be nonnegative.")
        if warmup_iters < 0:
            raise ValueError("'warmup_iters' must be nonnegative.")

        self.runner = runner
        self.monitor = monitor
        self.mode = mode
        self.max_no_improve_iters = max_no_improve_iters
        self.min_delta = min_delta
        self.warmup_iters = warmup_iters

        self.best_value: float | None = None
        self.no_improve_iters = 0
        self.observed_iters = 0


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self.best_value = None
        self.no_improve_iters = 0
        self.observed_iters = 0
        return True


    def _on_iteration_end(
        self,
        info: Mapping[str, Any],
        *args, **kwargs,
    ) -> bool:
        
        metric_name = resolve_metric_name(
            info=info,
            pattern=self.monitor,
            owner="Early-stopping",
        )
        value = scalar_value(info[metric_name])
        if value is None:
            raise TypeError(
                f"Early-stopping metric {metric_name!r} "
                "must be a numeric scalar."
            )
        if not math.isfinite(value):
            return False

        self.observed_iters += 1
        if self.observed_iters <= self.warmup_iters:
            return True

        if self.best_value is None or self._improved(value):
            self.best_value = value
            self.no_improve_iters = 0
            return True

        self.no_improve_iters += 1
        return (
            self.no_improve_iters
            < self.max_no_improve_iters
        )

    def _improved(self, value: float) -> bool:
        assert self.best_value is not None
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta


    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "best_value": self.best_value,
            "no_improve_iters": self.no_improve_iters,
            "observed_iters": self.observed_iters,
        }


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        best_value = state.get("best_value")
        if best_value is not None:
            best_value = float(best_value)
        no_improve_iters = state.get("no_improve_iters", 0)
        observed_iters = state.get("observed_iters", 0)
        if not isinstance(no_improve_iters, int):
            raise TypeError(
                "Early-stopping no-improvement counter must be an integer."
            )
        if not isinstance(observed_iters, int):
            raise TypeError(
                "Early-stopping observed counter must be an integer."
            )
        self.best_value = best_value
        self.no_improve_iters = no_improve_iters
        self.observed_iters = observed_iters
