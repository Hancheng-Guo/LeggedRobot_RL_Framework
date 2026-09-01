from pathlib import Path
from typing import Any

from app.utils.context import RuntimeContext
from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from utils.logging import close_handlers, configure_logging
from utils.scalar import scalar_metrics


class LoggingCallback(BaseCallback):

    def __init__(
        self,
        runner: BaseRunner,
        context: RuntimeContext,
        log_interval: int = 1,
        file_name: str = "training.log",
        console: bool = True,
        *args, **kwargs,
    ) -> None:
        
        if log_interval <= 0:
            raise ValueError("'log_interval' must be greater than 0.")
        if not file_name:
            raise ValueError("'file_name' cannot be empty.")

        self.runner = runner
        self.log_interval = log_interval
        self._closed = False
        self.logger, self._handlers = configure_logging(
            log_file=Path(context.save_dir) / "logs" / file_name,
            console=console,
        )


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self.logger.info("Training started.")
        return True


    def _on_iteration_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:

        iteration = info.get("runner/current_iter", None)
        if iteration is None:
            raise RuntimeError("Missing current_iter in runner.")
        iteration += 1
        if iteration % self.log_interval != 0:
            return True

        metrics = scalar_metrics(info)
        sorted_metrics = sorted(metrics.items())
        message_lines = [f"Iteration {iteration}"]
        if sorted_metrics:
            name_width = max(len(name) for name, _ in sorted_metrics)
            message_lines.extend(
                f"    {name:<{name_width}} : {value:>12.6g}"
                for name, value in sorted_metrics
            )
        message = "\n".join(message_lines)
        self.logger.info(message)
        return True


    def _on_train_end(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self.logger.info("Training ended.")
        self._flush()
        return True


    def _on_close(
        self,
        *args, **kwargs,
    ) -> bool:
        
        if self._closed:
            return True
        close_handlers(self.logger, self._handlers)
        self._closed = True
        return True


    def _flush(self) -> None:
        for handler in self._handlers:
            handler.flush()
