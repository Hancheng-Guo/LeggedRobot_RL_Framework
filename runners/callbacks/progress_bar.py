from typing import Any
from time import monotonic

from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback


class ProgressBarCallback(BaseCallback):

    def __init__(
        self,
        runner: BaseRunner,
        max_iterations: int,
        rollout_length: int,
        width: int = 30,
        refresh_interval: float = 0.2,
        *args, **kwargs,
    ) -> None:
        
        if max_iterations <= 0:
            raise ValueError("'max_iterations' must be greater than 0.")
        if rollout_length <= 0:
            raise ValueError("'rollout_length' must be greater than 0.")
        if width <= 0:
            raise ValueError("Progress bar 'width' must be greater than 0.")
        if refresh_interval < 0.0:
            raise ValueError("'refresh_interval' must be nonnegative.")

        self.runner = runner
        self.max_iterations = max_iterations
        self.rollout_length = rollout_length
        self.total_steps = max_iterations * rollout_length
        self.width = width
        self.refresh_interval = refresh_interval

        self._last_length = 0
        self._completed_iterations = 0
        self._completed_steps = 0
        self._cursor_position = 0
        self._last_refresh_time = 0.0


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self._last_length = 0
        self._completed_iterations = 0
        self._completed_steps = 0
        self._cursor_position = 0
        now = monotonic()
        self._last_refresh_time = now
        return True


    def _on_step_end(
        self,
        *args, **kwargs,
    ) -> bool:

        self._completed_steps += 1
        now = monotonic()
        if now - self._last_refresh_time < self.refresh_interval:
            return True
        
        self._advance_cursor()
        self._render(now=now)

        return True


    def _on_iteration_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:
        
        current_iter = info.get("runner/current_iter", None)
        if current_iter is None:
            raise RuntimeError("Missing current_iter in runner.")
        
        self._completed_iterations = current_iter + 1
        self._clear()
        return True


    def _on_train_end(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self._clear()
        self._last_length = 0
        return True


    def _render(
        self,
        now: float,
    ) -> None:
        
        completed = min(
            self._completed_iterations,
            self.max_iterations,
        )
        fraction = min(self._completed_steps, self.total_steps) / self.total_steps
        filled = round(self.width * fraction)
        bar = self._build_bar(filled)
        message = (
            f"\r[{bar}] {completed}/{self.max_iterations} "
            f"({fraction:6.2%})"
        )
        padding = " " * max(0, self._last_length - len(message))
        print(message + padding, end="", flush=True)
        self._last_length = len(message)
        self._last_refresh_time = now


    def _clear(self) -> None:

        if self._last_length == 0:
            return
        
        print(
            "\r" + " " * self._last_length + "\r",
            end="",
            flush=True,
        )
        self._last_length = 0


    def _advance_cursor(self) -> None:
        self._cursor_position = (
            self._cursor_position + 1
        ) % self.width


    def _build_bar(
        self,
        filled: int
    ) -> str:
        
        bar_parts = [
            "#" if index < filled else "-"
            for index in range(self.width)
        ]
        if self._cursor_position >= filled:
            bar_parts[self._cursor_position] = ">"

        return "".join(bar_parts)
