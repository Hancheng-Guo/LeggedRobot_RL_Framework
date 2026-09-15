from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from app.utils.context import RuntimeContext
from runners.callbacks.stage import StageCallback
from runners.base import BaseRunner
from runners.registry import RUNNER_TYPE_MAP
from utils.component import create_component, Component
from utils.config import load_yaml
from utils.logging import get_logger


logger = get_logger(__name__)


class StageTrainResult(Enum):
    STAGE_COMPLETED = auto()
    STAGE_ALREADY_COMPLETED = auto()
    STOPPED_BY_CALLBACK = auto()
    MAX_ITERATIONS_REACHED = auto()


@dataclass(frozen=True, slots=True)
class StageResumeInfo:
    checkpoint_path: Path
    stage_completed: bool


class StageManager:

    def __init__(
        self,
        component: dict,
        context: RuntimeContext,
        stage_detail: list[dict],
        load_dir: Path,
    ) -> None:
        
        self.component = component
        self.context = context
        self.stage_detail = stage_detail
        self.load_dir = load_dir

        self.runner: BaseRunner
        self.current_stage: int
        self.max_stage: int
        
        self._setup()


    def _setup(self) -> None:

        self.current_stage = 0
        self.max_stage = len(self.stage_detail) - 1

        if not self.stage_detail:
            raise ValueError("Stage config cannot be empty.")

        for stage in self.stage_detail:
            self._parse_stage(stage)

        for name, detail in self.component.items():
            if len(detail) - 1 > self.max_stage:
                raise ValueError(
                    f"{name} config size does not match stages."
                )


    @staticmethod
    def _parse_stage(
        stage: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:

        if not isinstance(stage, Mapping) or len(stage) != 1:
            raise ValueError(
                "Each stage must contain exactly one stage name."
            )

        stage_name, raw_config = next(iter(stage.items()))
        if not isinstance(stage_name, str) or not stage_name:
            raise ValueError("Stage name must be a non-empty string.")
        if not isinstance(raw_config, Mapping):
            raise TypeError(
                f"Stage {stage_name!r} configuration must be a mapping."
            )

        expected_keys = {"max_iterations", "transition"}
        if set(raw_config) != expected_keys:
            raise ValueError(
                f"Stage {stage_name!r} must contain exactly: "
                "max_iterations, transition."
            )

        max_iterations = raw_config["max_iterations"]
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError(
                f"Stage {stage_name!r} max_iterations must be a "
                "positive integer."
            )

        transition = raw_config["transition"]
        if not isinstance(transition, list) or not transition:
            raise ValueError(
                f"Stage {stage_name!r} transition must be a non-empty list."
            )

        return stage_name, dict(raw_config)


    def train(self) -> None:

        resume_info = self._prepare_training_stage()
        while self.continue_training:
            train_result = self._train_current(resume_info)
            resume_info = None

            if train_result is StageTrainResult.STAGE_ALREADY_COMPLETED:
                self.current_stage += 1
                continue

            if train_result is StageTrainResult.STAGE_COMPLETED:
                self._save_completed_stage()
                self.current_stage += 1
                continue

            if train_result is StageTrainResult.STOPPED_BY_CALLBACK:
                for callback in self.runner.stop_callback:
                    logger.warning(
                        f"Training was stopped by callback "
                        f"{type(callback).__name__!r} during stage "
                        f"{self.current_stage}."
                    )
            else:   # StageTrainResult.MAX_ITERATIONS_REACHED
                logger.warning(f"Stage {self.current_stage} timeout.")
            break


    @property
    def continue_training(self) -> bool:

        self._validate_current_stage()
        return self.current_stage < len(self.stage_detail)


    def _validate_current_stage(self) -> None:

        if not 0 <= self.current_stage <= len(self.stage_detail):
            raise RuntimeError(
                f"Invalid current stage index: {self.current_stage}."
            )


    def _get_effective_stage(self) -> int:

        if self.continue_training:
            return self.current_stage
        return len(self.stage_detail) - 1


    def _train_current(
        self,
        resume_info: StageResumeInfo | None = None,
    ) -> StageTrainResult:

        if not self.continue_training:
            raise RuntimeError("All stages have already been completed.")

        current_component = self._get_current_component()
        current_stage_callback = self._build_current_stage_callback()
        checkpoint_path = (
            resume_info.checkpoint_path
            if resume_info is not None
            else None
        )

        self._build_runner(
            component=current_component,
            stage_callback=current_stage_callback,
            max_iterations=self._get_current_max_iterations(),
            stage_index=self.current_stage,
            checkpoint_path=checkpoint_path,
        )
        if checkpoint_path is not None:
            self.runner.load(load_optimizer=True)
            if resume_info is not None and resume_info.stage_completed:
                return StageTrainResult.STAGE_ALREADY_COMPLETED
        self.runner.train()

        if current_stage_callback.stop_training:
            return StageTrainResult.STAGE_COMPLETED
        if self.runner.stop_callback:
            return StageTrainResult.STOPPED_BY_CALLBACK
        return StageTrainResult.MAX_ITERATIONS_REACHED


    def _prepare_training_stage(self) -> StageResumeInfo | None:

        if hasattr(self, "runner"):
            return None
        if self.load_dir.resolve() != Path(self.context.save_dir).resolve():
            return None

        checkpoint_root = self.load_dir / "checkpoints"
        candidates: list[tuple[int, Path]] = []
        for path in checkpoint_root.glob("stage_*/latest.pt"):
            try:
                stage_index = int(path.parent.name.removeprefix("stage_"))
            except ValueError:
                continue
            if 0 <= stage_index < len(self.stage_detail):
                candidates.append((stage_index, path))

        if not candidates:
            raise FileNotFoundError(
                "No training checkpoint was found in "
                f"{checkpoint_root!s}. Expected stage_*/latest.pt."
            )

        stage_index, checkpoint_path = max(candidates)
        self.current_stage = stage_index
        return StageResumeInfo(
            checkpoint_path=checkpoint_path,
            stage_completed=(
                checkpoint_path.parent / "stage_completed"
            ).is_file(),
        )


    def _save_completed_stage(self) -> None:

        checkpoint_path = self.runner.save()
        marker_path = checkpoint_path.parent / "stage_completed"
        temporary_path = marker_path.with_suffix(".tmp")
        try:
            temporary_path.write_text(
                str(self.current_stage),
                encoding="utf-8",
            )
            temporary_path.replace(marker_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


    def _get_current_component(self) -> Component:

        current_component = dict()
        stage = self._get_effective_stage()
        full_configuration = not hasattr(self, "runner")

        for name, detail in self.component.items():

            if full_configuration:
                current_component[name] = detail[min(stage, len(detail) - 1)]
                continue

            if stage <= len(detail) - 1:

                if stage >= 1 and detail[stage] == detail[stage - 1]:
                    continue
            
                current_component[name] = detail[stage]

        return create_component(current_component, self.load_dir)


    def _build_current_stage_callback(self) -> StageCallback:

        _, stage_config = self._parse_stage(
            self.stage_detail[self.current_stage]
        )
        return StageCallback(
            condition=stage_config["transition"]
        )


    def _get_current_max_iterations(self) -> int:

        stage = self._get_effective_stage()
        _, stage_config = self._parse_stage(self.stage_detail[stage])
        return stage_config["max_iterations"]


    def _build_runner(
        self,
        component: Component,
        max_iterations: int,
        stage_callback: StageCallback | None = None,
        stage_index: int | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:

        if component.runner is None:
            if not hasattr(self, "runner"):
                raise RuntimeError(
                    f"runner instance is required."
                )
            self.runner.config_update(
                component=component,
                max_iterations=max_iterations,
                stage_index=stage_index,
            )

        else:
            runner_type_name = component.runner.type
            if runner_type_name not in RUNNER_TYPE_MAP:
                raise ValueError(
                    f"Invalid runner type: {runner_type_name!r}. "
                )

            runner_type = RUNNER_TYPE_MAP[runner_type_name]
            runner_config = load_yaml(component.runner.config)

            if (
                not hasattr(self, "runner")
                or not isinstance(self.runner, runner_type)
            ):
                self.runner = runner_type(
                    context=self.context
                )
            if checkpoint_path is not None:
                self.runner.prepare_checkpoint_load(checkpoint_path)
            self.runner.config_update(
                component=component,
                max_iterations=max_iterations,
                stage_index=stage_index,
                **runner_config
            )

        if stage_callback is not None:
            stage_callback.set_runner(self.runner)
        self.runner.stage_update(stage_callback)
        

    def test(
        self,
        *args, **kwargs
    ) -> None:

        if hasattr(self, "runner"):
            if self.continue_training:
                logger.warning("Model is not trained completely.")
            self.runner.stage_update(None)
            self.runner.test(*args, **kwargs)
            return

        checkpoint_path = self._prepare_evaluation_stage()
        if self.continue_training:
            logger.warning("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component,
            max_iterations=self._get_current_max_iterations(),
            stage_index=self._get_effective_stage(),
            checkpoint_path=checkpoint_path,
        )
        if checkpoint_path is not None:
            self._load_evaluation_checkpoint()
        self.runner.test(*args, **kwargs)


    def play(
        self,
        *args, **kwargs
    ) -> None:

        if hasattr(self, "runner"):
            if self.continue_training:
                logger.warning("Model is not trained completely.")
            self.runner.stage_update(None)
            self.runner.play(*args, **kwargs)
            return

        checkpoint_path = self._prepare_evaluation_stage()
        if self.continue_training:
            logger.warning("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component,
            max_iterations=self._get_current_max_iterations(),
            stage_index=self._get_effective_stage(),
            checkpoint_path=checkpoint_path,
        )
        if checkpoint_path is not None:
            self._load_evaluation_checkpoint()
        self.runner.play(*args, **kwargs)


    def _prepare_evaluation_stage(self) -> Path | None:

        if hasattr(self, "runner"):
            return None
        if self.load_dir.resolve() != Path(self.context.save_dir).resolve():
            return None

        checkpoint_root = self.load_dir / "checkpoints"
        candidates: list[tuple[int, Path]] = []
        for path in checkpoint_root.glob("stage_*/latest.pt"):
            try:
                stage_index = int(path.parent.name.removeprefix("stage_"))
            except ValueError:
                continue
            if 0 <= stage_index < len(self.stage_detail):
                candidates.append((stage_index, path))
        if candidates:
            stage_index, checkpoint_path = max(candidates)
            self.current_stage = stage_index
            return checkpoint_path

        checkpoint_path = checkpoint_root / "latest.pt"
        if checkpoint_path.is_file():
            return checkpoint_path
        raise FileNotFoundError(
            "No evaluation checkpoint was found in "
            f"{checkpoint_root!s}. Expected stage_*/latest.pt or latest.pt."
        )


    def _load_evaluation_checkpoint(
        self,
    ) -> None:
        
        if not hasattr(self, "runner"):
            raise RuntimeError("Runner was not built for checkpoint loading.")
        self.runner.load(load_optimizer=False)


    def save(self) -> Path:
        
        if not hasattr(self, "runner"):
            raise RuntimeError("Cannot save before a runner has been built.")
        return self.runner.save()


    def close(self) -> None:

        if hasattr(self, "runner"):
            self.runner.close()
            del self.runner
