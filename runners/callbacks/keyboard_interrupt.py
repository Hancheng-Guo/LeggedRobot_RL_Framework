import os
import sys
from importlib import import_module
from typing import Any

from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from utils.logging import get_logger


logger = get_logger(__name__)
STOP_KEY = "\x18"  # Ctrl+X


class KeyboardInterruptCallback(BaseCallback):
    """Stop training with Ctrl+X after completing the current update."""

    def __init__(
        self,
        runner: BaseRunner,
        *args, **kwargs
    ) -> None:
        
        self.runner = runner
        self._stop_requested = False
        self._completed_iteration = False
        self._terminal_settings: Any = None


    def _on_train_start(
        self,
        *args, **kwargs
    ) -> bool:

        self._stop_requested = False
        self._completed_iteration = False
        if not sys.stdin.isatty():
            logger.warning("Keyboard stop is unavailable without an interactive terminal.")
            return True
        if os.name != "nt":
            termios = import_module("termios")
            tty = import_module("tty")
            self._terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        logger.info("Press Ctrl+X to stop training after the current update.")
        return True


    def _check_key(self) -> None:

        if self._stop_requested or not sys.stdin.isatty():
            return
        if os.name == "nt":
            import msvcrt

            key = msvcrt.getwch() if msvcrt.kbhit() else None
        else:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], 0)
            key = sys.stdin.read(1) if ready else None
        if key == STOP_KEY:
            self._stop_requested = True
            print(flush=True)
            logger.warning("Ctrl+X received; finishing the current update before stopping.")


    def _on_step_end(self, *args, **kwargs) -> bool:
        self._check_key()
        return not self._stop_requested


    def _on_iteration_end(self, *args, **kwargs) -> bool:
        self._completed_iteration = True
        self._check_key()
        return not self._stop_requested


    def _on_train_end(self, *args, **kwargs) -> bool:
        self._restore_terminal()
        if self._stop_requested and self._completed_iteration:
            checkpoint_path = self.runner.save()
            logger.info("Interrupted training checkpoint saved to %s.", checkpoint_path.as_posix())
        return True

    def _on_close(self, *args, **kwargs) -> bool:
        self._restore_terminal()
        return True


    def _restore_terminal(self) -> None:
        if self._terminal_settings is not None:
            termios = import_module("termios")
            termios.tcsetattr(
                sys.stdin.fileno(), termios.TCSADRAIN, self._terminal_settings
            )
            self._terminal_settings = None
