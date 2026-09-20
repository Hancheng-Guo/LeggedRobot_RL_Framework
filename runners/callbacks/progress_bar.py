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
        self._training = False
        self._mode: str | None = None
        self._play_total_steps = 0
        self._test_completed_episodes = 0
        self._test_total_episodes = 0


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self._training = True
        self._mode = "train"
        self._last_length = 0
        self._completed_iterations = self.runner.current_iteration + 1
        self._completed_steps = (
            self._completed_iterations * self.rollout_length
        )
        self._cursor_position = 0
        now = monotonic()
        self._last_refresh_time = now
        return True


    def _on_step_end(
        self,
        completed_episodes: int | None = None,
        total_episodes: int | None = None,
        *args, **kwargs,
    ) -> bool:

        if self._mode is None:
            return True

        self._completed_steps += 1
        if completed_episodes is not None:
            self._test_completed_episodes = completed_episodes
        if total_episodes is not None:
            self._test_total_episodes = total_episodes
        now = monotonic()
        if now - self._last_refresh_time < self.refresh_interval:
            return True
        
        self._advance_cursor()
        if self._mode == "train":
            self._render_train(now=now)
        elif self._mode == "play":
            self._render_play(now=now)
        else:
            self._render_test(now=now)

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
        self._training = False
        self._mode = None
        self._last_length = 0
        return True


    def _on_play_start(
        self,
        num_steps: int,
        *args, **kwargs,
    ) -> bool:
        self._start_evaluation(mode="play")
        self._play_total_steps = num_steps
        return True


    def _on_play_end(self, *args, **kwargs) -> bool:
        self._finish_evaluation()
        return True


    def _on_test_start(
        self,
        num_episodes: int,
        *args, **kwargs,
    ) -> bool:
        self._start_evaluation(mode="test")
        self._test_total_episodes = num_episodes
        self._test_completed_episodes = 0
        return True


    def _on_test_end(self, *args, **kwargs) -> bool:
        self._finish_evaluation()
        return True


    def _start_evaluation(self, mode: str) -> None:
        self._clear()
        self._mode = mode
        self._training = False
        self._completed_steps = 0
        self._cursor_position = 0
        self._last_refresh_time = monotonic()


    def _finish_evaluation(self) -> None:
        self._clear()
        self._mode = None
        self._last_length = 0


    def _render_train(
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
            f"\r[{bar}] {completed}/{self.max_iterations} iterations "
            f"({fraction:6.2%})"
        )
        padding = " " * max(0, self._last_length - len(message))
        print(message + padding, end="", flush=True)
        self._last_length = len(message)
        self._last_refresh_time = now


    def _render_play(self, now: float) -> None:
        completed = min(self._completed_steps, self._play_total_steps)
        fraction = completed / self._play_total_steps
        filled = round(self.width * fraction)
        bar = self._build_moving_bar(filled)
        self._print_progress(
            f"\r[{bar}] {completed}/{self._play_total_steps} steps "
            f"({fraction:6.2%})",
            now,
        )


    def _render_test(self, now: float) -> None:
        fraction = min(
            self._test_completed_episodes,
            self._test_total_episodes,
        ) / self._test_total_episodes
        filled = round(self.width * fraction)
        bar = self._build_moving_bar(filled)
        self._print_progress(
            f"\r[{bar}] {self._test_completed_episodes}/"
            f"{self._test_total_episodes} episodes "
            f"({fraction:6.2%})",
            now,
        )


    def _print_progress(self, message: str, now: float) -> None:
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


    def _build_moving_bar(self, filled: int) -> str:
        filled = min(max(filled, 0), self.width)
        bar_parts = [
            "#" if index < filled else "-"
            for index in range(self.width)
        ]
        bar_parts[self._cursor_position] = ">"
        return "".join(bar_parts)
