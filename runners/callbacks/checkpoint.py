from pathlib import Path
from typing import Any

from app.utils.context import RuntimeContext
from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback


def _stage_directory_name(
    stage_index: int | None,
) -> str | None:

    if stage_index is None:
        return None
    return f"stage_{stage_index:03d}"


class CheckpointCallback(BaseCallback):

    def __init__(
        self,
        runner: BaseRunner,
        context: RuntimeContext,
        save_iter_interval: int = 100,
        directory_name: str = "checkpoints",
        save_on_train_end: bool = False,
        stage_index: int | None = None,
        *args, **kwargs,
    ) -> None:

        if save_iter_interval <= 0:
            raise ValueError("'save_iter_interval' must be greater than 0.")
        if not directory_name:
            raise ValueError("'directory_name' cannot be empty.")

        self.runner = runner
        self.save_iter_interval = save_iter_interval
        self.save_on_train_end = save_on_train_end
        self.stage_index = stage_index

        self.checkpoint_dir = Path(context.save_dir) / directory_name
        stage_directory_name = _stage_directory_name(stage_index)
        if stage_directory_name is not None:
            self.checkpoint_dir = self.checkpoint_dir / stage_directory_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_iteration = 0
        self.last_saved_iteration = 0


    def _on_train_start(
        self,
        *args, **kwargs
    ) -> bool:

        completed_iterations = self.runner.current_iteration + 1
        self.global_iteration = completed_iterations
        self.last_saved_iteration = self.global_iteration
        return True


    def _on_iteration_end(
        self,
        info: dict[str, Any],
        *args, **kwargs,
    ) -> bool:

        self.global_iteration += 1
        return True


    def _on_iteration_finalize(
        self,
        *args, **kwargs,
    ) -> bool:

        if self.global_iteration % self.save_iter_interval == 0:
            self._save()
        return True


    def _on_train_end(
        self,
        info: dict[str, Any] | None = None,
        *args, **kwargs,
    ) -> bool:

        if (
            self.save_on_train_end
            and self.global_iteration > 0
            and self.last_saved_iteration != self.global_iteration
        ):
            self._save()
        return True


    def _save(self) -> None:

        checkpoint_path = self.checkpoint_dir / (
            f"checkpoint_{self.global_iteration:08d}.pt"
        )
        self.runner.save(
            path=checkpoint_path,
        )
        self.runner.save(
            path=self.checkpoint_dir / "latest.pt",
        )
        self.last_saved_iteration = self.global_iteration
