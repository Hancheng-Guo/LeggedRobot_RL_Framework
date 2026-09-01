import torch
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
from tensorboard.program import TensorBoard
from torch.utils.tensorboard import SummaryWriter

from app.utils.context import RuntimeContext
from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from utils.logging import get_logger
from utils.scalar import scalar_metrics


logger = get_logger(__name__)


class TensorboardCallback(BaseCallback):

    _SERVER_URLS: dict[Path, str] = {}

    _METRIC_REDUCTIONS = frozenset((
        "value",
        "mean",
        "std",
        "min",
        "max",
        "histogram",
    ))

    def __init__(
        self,
        runner: BaseRunner,
        context: RuntimeContext,
        log_dir: str = "tensorboard",
        step_log_interval: int = 1,
        histogram_log_interval: int = 100,
        step_metrics: Mapping[str, Sequence[str]] | None = None,
        stage_index: int | None = None,
        *args, **kwargs,
    ) -> None:
        
        if step_log_interval <= 0:
            raise ValueError("'step_log_interval' must be greater than 0.")
        if histogram_log_interval <= 0:
            raise ValueError(
                "'histogram_log_interval' must be greater than 0."
            )
        if not log_dir:
            raise ValueError("'log_dir_name' cannot be empty.")

        self.runner = runner
        self.step_log_interval = step_log_interval
        self.histogram_log_interval = histogram_log_interval
        self.step_metrics = self._validate_step_metrics(step_metrics)
        self.tensorboard_root_dir = Path(context.save_dir) / log_dir
        self.tensorboard_log_dir = (
            self.tensorboard_root_dir
            if stage_index is None
            else self.tensorboard_root_dir / f"stage_{stage_index:03d}"
        )

        self.writer = SummaryWriter(
            log_dir=str((self.tensorboard_log_dir).resolve())
        )
        
        self.global_step = 0
        self.global_iteration = 0
        self.tensorboard_url: str
        self._tensorboard: TensorBoard
        self._server_started = False
        self._closed = False


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:

        if not self._server_started:
            if self.tensorboard_root_dir not in self._SERVER_URLS:
                self._tensorboard = TensorBoard()
                self._tensorboard.configure(
                    argv=(
                        "tensorboard",
                        "--logdir",
                        str((self.tensorboard_root_dir).resolve()),
                        "--host",
                        "0.0.0.0",
                    )
                )
                self._SERVER_URLS[self.tensorboard_root_dir] = (
                    self._tensorboard.launch()
                )
            self.tensorboard_url = self._SERVER_URLS[self.tensorboard_root_dir]
            self._server_started = True

        logger.info(f"TensorBoard dir: {self.tensorboard_log_dir}")
        logger.info(f"TensorBoard url: {self.tensorboard_url}")
        return True


    def _on_step_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:
        
        self.global_step += 1
        if self.global_step % self.step_log_interval == 0:
            self._log_step_metrics(info)
        return True


    def _on_iteration_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:
        
        self.global_iteration += 1
        for name, value in scalar_metrics(info).items():
            self.writer.add_scalar(name, value, self.global_iteration)
        return True


    def _on_train_end(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self.writer.flush()
        return True


    def _on_close(
        self,
        *args, **kwargs,
    ) -> bool:
        
        if not self._closed:
            self.writer.close()
            self._closed = True
        return True


    def _log_step_metrics(
        self,
        info: Mapping[str, Any],
    ) -> None:

        scalar_values = scalar_metrics(info)
        for name, value in info.items():
            reductions = self._matching_reductions(name)
            if not reductions:
                continue

            if name in scalar_values:
                if "value" in reductions:
                    self.writer.add_scalar(
                        name,
                        scalar_values[name],
                        self.global_step,
                    )
                continue

            if not isinstance(value, torch.Tensor):
                continue

            if "value" in reductions:
                raise ValueError(
                    f"Vector metric {name!r} cannot use the 'value' "
                    "reduction."
                )

            tensor = value.detach().float()
            for reduction in reductions - {"histogram"}:
                reduced = self._reduce_tensor(tensor, reduction)
                self.writer.add_scalar(
                    f"{name}/{reduction}",
                    reduced,
                    self.global_step,
                )

            if (
                "histogram" in reductions
                and self.global_step % self.histogram_log_interval == 0
            ):
                self.writer.add_histogram(
                    f"{name}/distribution",
                    tensor,
                    self.global_step,
                )


    def _matching_reductions(
        self,
        name: str
    ) -> set[str]:
        
        return {
            reduction
            for pattern, reductions in self.step_metrics.items()
            if fnmatchcase(name, pattern)
            for reduction in reductions
        }


    @staticmethod
    def _reduce_tensor(tensor: torch.Tensor, reduction: str) -> float:
        if reduction == "mean":
            result = tensor.mean()
        elif reduction == "std":
            result = tensor.std(correction=0)
        elif reduction == "min":
            result = tensor.min()
        elif reduction == "max":
            result = tensor.max()
        else:
            raise ValueError(f"Unsupported vector reduction: {reduction!r}.")
        return float(result.item())


    @classmethod
    def _validate_step_metrics(
        cls,
        step_metrics: Mapping[str, Sequence[str]] | None,
    ) -> dict[str, frozenset[str]]:
        
        if step_metrics is None:
            return {}

        validated: dict[str, frozenset[str]] = {}
        for pattern, reductions in step_metrics.items():

            if not isinstance(pattern, str) or not pattern:
                raise ValueError("Step metric patterns must be non-empty strings.")
            if isinstance(reductions, str):
                raise TypeError(
                    f"Step metric reductions for {pattern!r} must be a sequence."
                )
            
            reduction_set = frozenset(reductions)
            unknown = reduction_set - cls._METRIC_REDUCTIONS
            if unknown:
                raise ValueError(
                    f"Unsupported metric reductions for {pattern!r}: "
                    f"{sorted(unknown)}."
                )
            
            validated[pattern] = reduction_set

        return validated
