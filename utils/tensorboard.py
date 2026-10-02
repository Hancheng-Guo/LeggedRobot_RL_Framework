"""Shared TensorBoard launch and initial-load display."""

import logging
import threading
from pathlib import Path
from collections.abc import Callable
from tensorboard.program import TensorBoard

from utils import get_logger


logger = get_logger(__name__)


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


def launch_tensorboard(
    log_dir: Path,
    *,
    host: str = "127.0.0.1",
    initial_load_timeout: float = 600.0,
    max_reload_threads: int = 4,
    tensorboard_factory: Callable[[], TensorBoard] = TensorBoard,
) -> str:

    tensorboard = tensorboard_factory()
    tensorboard.configure(
        argv=(
            "tensorboard",
            "--logdir",
            str(log_dir.resolve()),
            "--host",
            host,
            "--load_fast",
            "false",
            "--max_reload_threads",
            str(max_reload_threads),
        )
    )
    console_handler = next(
        (
            handler
            for handler in logging.getLogger("rl_framework").handlers
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and getattr(handler.stream, "isatty", lambda: False)()
        ),
        None,
    )
    if console_handler is None:
        logger.info("Waiting for TensorBoard to load existing event data.")
    else:
        original_terminator = console_handler.terminator
        console_handler.terminator = ""
        try:
            logger.info("Waiting for TensorBoard to load existing event data. ")
        finally:
            console_handler.terminator = original_terminator

    completed = threading.Event()
    handler = _TensorboardLoadHandler(completed)
    tensorboard_logger = logging.getLogger("tensorboard")
    previous_level = tensorboard_logger.level
    tensorboard_logger.setLevel(logging.INFO)
    tensorboard_logger.addHandler(handler)
    stop_spinner = threading.Event()
    spinner: threading.Thread | None = None
    if console_handler is not None:
        def animate() -> None:
            while not stop_spinner.is_set():
                for frame in "-/|\\":
                    if stop_spinner.is_set():
                        break
                    console_handler.acquire()
                    try:
                        console_handler.stream.write(frame)
                        console_handler.flush()
                    finally:
                        console_handler.release()
                    stop_spinner.wait(0.15)
                    console_handler.acquire()
                    try:
                        console_handler.stream.write("\b \b")
                        console_handler.flush()
                    finally:
                        console_handler.release()

        spinner = threading.Thread(target=animate, daemon=True)
        spinner.start()
    try:
        url = tensorboard.launch()
        if not completed.wait(initial_load_timeout):
            raise TimeoutError(
                "TensorBoard did not finish its initial event-data load "
                f"within {initial_load_timeout:g} seconds."
            )
        return url
    finally:
        stop_spinner.set()
        if spinner is not None and console_handler is not None:
            spinner.join()
            console_handler.acquire()
            try:
                console_handler.stream.write("\n")
                console_handler.flush()
            finally:
                console_handler.release()
        tensorboard_logger.removeHandler(handler)
        tensorboard_logger.setLevel(previous_level)
