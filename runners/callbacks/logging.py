from typing import Any

from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from utils.logging import get_logger
from utils.scalar import scalar_metrics


class LoggingCallback(BaseCallback):

    def __init__(
        self,
        runner: BaseRunner,
        log_interval: int = 1,
        *args, **kwargs,
    ) -> None:
        
        if log_interval <= 0:
            raise ValueError("'log_interval' must be greater than 0.")
        self.runner = runner
        self.log_interval = log_interval
        self.logger = get_logger()


    def _stage_label(self) -> str:
        stage_index = self.runner.stage_index
        return "evaluation" if stage_index is None else f"stage {stage_index}"


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:
        
        self.logger.info(f"Training started for {self._stage_label()}.")
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
        message_lines = [
            f"Stage {self.runner.stage_index}, iteration {iteration}"
            if self.runner.stage_index is not None
            else f"Iteration {iteration}"
        ]
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
        
        self.logger.info(f"Training ended for {self._stage_label()}.")
        return True


    def _on_test_start(self, *args, **kwargs) -> bool:
        self.logger.info(f"Testing started for {self._stage_label()}.")
        return True


    def _on_test_end(
        self,
        info: dict[str, Any] | None = None,
        *args, **kwargs,
    ) -> bool:
        if info is None:
            self.logger.info(f"Testing ended for {self._stage_label()}.")
        else:
            self.logger.info(
                f"Testing ended for {self._stage_label()}: "
                f"{info['num_episodes']} episodes, "
                f"mean reward {info['mean_reward']:.6g}, "
                f"mean episode length {info['mean_episode_length']:.6g}."
            )
        return True


    def _on_play_start(self, *args, **kwargs) -> bool:
        self.logger.info(f"Playback started for {self._stage_label()}.")
        return True


    def _on_play_end(
        self,
        info: dict[str, Any] | None = None,
        *args, **kwargs,
    ) -> bool:
        output_paths = () if info is None else info.get("output_paths", ())
        if output_paths:
            self.logger.info(
                "Playback ended. Saved output to: %s",
                ", ".join(str(path) for path in output_paths),
            )
        else:
            self.logger.warning(
                "Playback ended without saving output because no rendered "
                "frames were produced."
            )
        return True
