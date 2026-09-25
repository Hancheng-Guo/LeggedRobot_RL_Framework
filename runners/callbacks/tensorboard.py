import logging
import threading
import torch
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from tensorboard.program import TensorBoard
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from app.utils.context import RuntimeContext
from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from utils.logging import get_logger
from utils.scalar import scalar_metrics


logger = get_logger(__name__)


@dataclass
class _HistogramAccumulator:
    bin_edges: torch.Tensor
    counts: torch.Tensor
    num: torch.Tensor
    total: torch.Tensor
    sum_squares: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor

    @classmethod
    def create(
        cls,
        values: torch.Tensor,
        bin_edges: Sequence[float],
    ) -> "_HistogramAccumulator":

        edges = torch.as_tensor(
            bin_edges,
            dtype=values.dtype,
            device=values.device,
        )
        return cls(
            bin_edges=edges,
            counts=torch.zeros(
                edges.numel() - 1,
                dtype=torch.long,
                device=values.device,
            ),
            num=torch.zeros((), dtype=torch.long, device=values.device),
            total=torch.zeros((), dtype=values.dtype, device=values.device),
            sum_squares=torch.zeros(
                (), dtype=values.dtype, device=values.device
            ),
            minimum=torch.full(
                (), torch.inf, dtype=values.dtype, device=values.device
            ),
            maximum=torch.full(
                (), -torch.inf, dtype=values.dtype, device=values.device
            ),
        )

    def update(
        self,
        values: torch.Tensor
    ) -> None:

        flat_values = values.flatten()
        if flat_values.numel() == 0:
            return
        bucket_ids = torch.bucketize(
            flat_values,
            self.bin_edges[1:-1],
        )
        self.counts += torch.bincount(
            bucket_ids,
            minlength=self.counts.numel(),
        )
        self.num += flat_values.numel()
        self.total += flat_values.sum()
        self.sum_squares += torch.square(flat_values).sum()
        self.minimum.copy_(torch.minimum(self.minimum, flat_values.min()))
        self.maximum.copy_(torch.maximum(self.maximum, flat_values.max()))


class _TensorboardLoadHandler(logging.Handler):

    def __init__(
        self,
        completed: threading.Event
    ) -> None:

        super().__init__()

        self.completed = completed


    def emit(
        self,
        record: logging.LogRecord
    ) -> None:

        if record.getMessage().startswith("TensorBoard done reloading."):
            self.completed.set()


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
        histogram_step_interval: int = 2000,
        flush_secs: int = 10,
        initial_load_timeout: float = 600.0,
        max_reload_threads: int = 4,
        step_metrics: Mapping[str, Sequence[str]] | None = None,
        stage_index: int | None = None,
        *args, **kwargs,
    ) -> None:
        
        if step_log_interval <= 0:
            raise ValueError("'step_log_interval' must be greater than 0.")
        if histogram_step_interval <= 0:
            raise ValueError(
                "'histogram_step_interval' must be greater than 0."
            )
        if (
            not isinstance(flush_secs, int)
            or isinstance(flush_secs, bool)
            or flush_secs <= 0
        ):
            raise ValueError("'flush_secs' must be a positive integer.")
        if not log_dir:
            raise ValueError("'log_dir_name' cannot be empty.")
        if initial_load_timeout <= 0.0:
            raise ValueError("'initial_load_timeout' must be greater than 0.")
        if (
            not isinstance(max_reload_threads, int)
            or isinstance(max_reload_threads, bool)
            or max_reload_threads <= 0
        ):
            raise ValueError("'max_reload_threads' must be a positive integer.")

        self.runner = runner
        self.step_log_interval = step_log_interval
        self.histogram_step_interval = histogram_step_interval
        self.flush_secs = flush_secs
        self.initial_load_timeout = float(initial_load_timeout)
        self.max_reload_threads = max_reload_threads
        self.step_metrics = self._validate_step_metrics(step_metrics)
        self.tensorboard_root_dir = Path(context.save_dir) / log_dir
        self.tensorboard_log_dir = (
            self.tensorboard_root_dir
            if stage_index is None
            else self.tensorboard_root_dir / f"stage_{stage_index:03d}"
        )

        self.writer: SummaryWriter | None = None
        self.global_step = 0
        self.global_iteration = 0
        self._pending_histograms: dict[str, _HistogramAccumulator] = {}
        self.tensorboard_url: str
        self._tensorboard: TensorBoard
        self._server_started = False
        self._closed = False


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:

        completed_iterations = self.runner.current_iteration + 1
        self.global_iteration = completed_iterations
        self.global_step = self.global_iteration * self.runner.rollout_length
        writer = self._ensure_writer(
            purge_step=(
                self.global_step + 1
                if completed_iterations > 0
                else None
            )
        )
        writer.flush()

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
                        "--load_fast",
                        "false",
                        "--max_reload_threads",
                        str(self.max_reload_threads),
                    )
                )
                logger.info(
                    "Waiting for TensorBoard to load existing event data."
                )
                url = self._launch_after_initial_load()
                self._SERVER_URLS[self.tensorboard_root_dir] = url
            self.tensorboard_url = self._SERVER_URLS[self.tensorboard_root_dir]
            self._server_started = True

        logger.info("TensorBoard dir: %s", self.tensorboard_log_dir.as_posix())
        logger.info(f"TensorBoard url: {self.tensorboard_url}")
        return True


    def _launch_after_initial_load(self) -> str:

        completed = threading.Event()
        handler = _TensorboardLoadHandler(completed)
        tensorboard_logger = logging.getLogger("tensorboard")
        previous_level = tensorboard_logger.level
        tensorboard_logger.setLevel(logging.INFO)
        tensorboard_logger.addHandler(handler)
        try:
            url = self._tensorboard.launch()
            if not completed.wait(self.initial_load_timeout):
                raise TimeoutError(
                    "TensorBoard did not finish its initial event-data load "
                    f"within {self.initial_load_timeout:g} seconds."
                )
            return url
        finally:
            tensorboard_logger.removeHandler(handler)
            tensorboard_logger.setLevel(previous_level)


    def _on_step_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:
        
        self.global_step += 1
        self._log_step_metrics(info)
        return True


    def _on_iteration_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:
        
        self.global_iteration += 1
        writer = self._ensure_writer()
        for name, value in scalar_metrics(info).items():
            writer.add_scalar(name, value, self.global_step)
        return True


    def _on_train_end(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self._ensure_writer().flush()
        return True


    def _on_close(
        self,
        *args, **kwargs,
    ) -> bool:
        
        if not self._closed:
            self._pending_histograms.clear()
            if self.writer is not None:
                self.writer.close()
            self._closed = True
        return True


    def _log_step_metrics(
        self,
        info: Mapping[str, Any],
    ) -> None:

        writer = self._ensure_writer()
        log = self.global_step % self.step_log_interval == 0

        scalar_values = scalar_metrics(info)
        for name, value in info.items():
            
            reductions = self._matching_reductions(name)
            if not reductions:
                continue

            if name in scalar_values:
                if log and "value" in reductions:
                    writer.add_scalar(
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
            if log:
                for reduction in reductions - {"histogram"}:
                    # Consume reusable step buffers synchronously. The
                    # reduction returns a Python float, so the writer never
                    # retains a reference to the source tensor.
                    reduced = self._reduce_tensor(tensor, reduction)
                    writer.add_scalar(
                        f"{name}/{reduction}",
                        reduced,
                        self.global_step,
                    )

            if "histogram" in reductions:
                # The accumulator copies counts and scalar statistics into
                # callback-owned tensors instead of retaining ``tensor``.
                self._accumulate_histogram(name, tensor)

        if self.global_step % self.histogram_step_interval == 0:
            self._flush_histograms()


    def _accumulate_histogram(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> None:

        accumulator = self._pending_histograms.get(name)
        if accumulator is None:
            writer = self._ensure_writer()
            accumulator = _HistogramAccumulator.create(
                tensor,
                writer.default_bins,
            )
            self._pending_histograms[name] = accumulator
        accumulator.update(tensor)


    def _flush_histograms(self) -> None:

        writer = self._ensure_writer()
        for name, accumulator in self._pending_histograms.items():
            if not int(accumulator.num.item()):
                continue
            counts = accumulator.counts.cpu()
            edges = accumulator.bin_edges.cpu()
            nonzero_bins = torch.nonzero(counts, as_tuple=False).flatten()
            first_bin = int(nonzero_bins[0].item())
            last_bin = int(nonzero_bins[-1].item())
            if first_bin > 0:
                bucket_counts = counts[first_bin - 1:last_bin + 1]
                bucket_limits = edges[first_bin:last_bin + 2]
            else:
                bucket_counts = torch.cat((
                    torch.zeros(1, dtype=counts.dtype),
                    counts[:last_bin + 1],
                ))
                bucket_limits = edges[:last_bin + 2]
            writer.add_histogram_raw(
                f"{name}/distribution",
                min=float(accumulator.minimum.item()),
                max=float(accumulator.maximum.item()),
                num=int(accumulator.num.item()),
                sum=float(accumulator.total.item()),
                sum_squares=float(accumulator.sum_squares.item()),
                bucket_limits=bucket_limits.tolist(),
                bucket_counts=bucket_counts.tolist(),
                global_step=self.global_step,
            )
        self._pending_histograms.clear()


    def _ensure_writer(
        self,
        purge_step: int | None = None,
    ) -> SummaryWriter:

        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir=str(self.tensorboard_log_dir.resolve()),
                purge_step=purge_step,
                flush_secs=self.flush_secs,
            )
        return self.writer


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
