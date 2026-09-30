from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from app.utils.run_workspace import CheckpointSelection
from runners.types import RestoreMode, TrainStopReason
from runners.base import BaseRunner
from runners.registry import RUNNER_TYPE_MAP
from utils.component import create_component, Component
from utils.config import load_yaml
from utils.logging import get_logger
from utils.runtime import RuntimeContext

if TYPE_CHECKING:
    from runners.utils.frames import VideoFormat, VideoFormats


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
        resume_selection: CheckpointSelection | None = None,
    ) -> None:
        
        self.component = component
        self.context = context
        self.stage_detail = stage_detail
        self.load_dir = load_dir
        self.resume_selection = resume_selection

        self.runner: BaseRunner
        self.current_stage: int
        self.max_stage: int
        
        self._setup()


    def use_fork_workspace(
        self,
        *,
        component: dict,
        context: RuntimeContext,
        load_dir: Path,
        resume_selection: CheckpointSelection,
    ) -> None:
        
        if hasattr(self, "runner"):
            raise RuntimeError("Cannot switch fork workspace while a runner is active.")
        self.component = component
        self.context = context
        self.load_dir = load_dir
        self.resume_selection = resume_selection


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
        logger.info("Training workflow started with %d stage(s).", len(self.stage_detail))
        while self.continue_training:
            train_result = self._train_current(resume_info)
            resume_info = None

            if train_result is StageTrainResult.STAGE_ALREADY_COMPLETED:
                logger.info("Stage %d was already completed; advancing.", self.current_stage)
                self.current_stage += 1
                continue

            if train_result is StageTrainResult.STAGE_COMPLETED:
                self._save_completed_stage()
                logger.info("Stage %d completed; advancing.", self.current_stage)
                self.current_stage += 1
                continue

            if train_result is StageTrainResult.STOPPED_BY_CALLBACK:
                logger.warning(
                    "Training was stopped by callback during stage %d.",
                    self.current_stage
                )
            else:   # StageTrainResult.MAX_ITERATIONS_REACHED
                logger.warning(f"Stage {self.current_stage} timeout.")
            break
        if not self.continue_training:
            logger.info("All training stages completed.")


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
        current_transition = self._get_current_transition()
        checkpoint_path = (
            resume_info.checkpoint_path
            if resume_info is not None
            else None
        )
        stage_name, _ = self._parse_stage(self.stage_detail[self.current_stage])
        logger.info(
            "Preparing stage %d (%s), max iterations=%d%s.",
            self.current_stage,
            stage_name,
            self._get_current_max_iterations(),
            f", checkpoint={checkpoint_path.as_posix()}" if checkpoint_path else "",
        )

        self._build_runner(
            component=current_component,
            transition=current_transition,
            max_iterations=self._get_current_max_iterations(),
            stage_index=self.current_stage,
            checkpoint_path=checkpoint_path,
        )
        if checkpoint_path is not None:
            self.runner.load(
                mode=RestoreMode.ADVANCE_STAGE
                if resume_info is not None and resume_info.stage_completed
                else RestoreMode.RESUME
            )
            logger.info(
                "Restored training checkpoint from %s.",
                checkpoint_path.as_posix()
            )
            if resume_info is not None and resume_info.stage_completed:
                return StageTrainResult.STAGE_ALREADY_COMPLETED
        result = self.runner.train()
        if result.reason is TrainStopReason.STAGE_COMPLETED:
            return StageTrainResult.STAGE_COMPLETED
        if result.reason is TrainStopReason.CALLBACK_STOPPED:
            return StageTrainResult.STOPPED_BY_CALLBACK
        return StageTrainResult.MAX_ITERATIONS_REACHED


    def _prepare_training_stage(self) -> StageResumeInfo | None:

        if hasattr(self, "runner"):
            return None

        selection = self.resume_selection
        if selection is None:
            return None

        self.current_stage = selection.stage_index
        return StageResumeInfo(selection.checkpoint, selection.stage_completed)


    def _save_completed_stage(self) -> None:
        checkpoint_path = self.runner.save()
        logger.info(
            "Saved completed stage %d checkpoint to %s.",
            self.current_stage,
            checkpoint_path.as_posix()
        )


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


    def _get_current_transition(self) -> list[dict]:
        _, stage_config = self._parse_stage(self.stage_detail[self.current_stage])
        return stage_config["transition"]


    def _get_current_max_iterations(self) -> int:

        stage = self._get_effective_stage()
        _, stage_config = self._parse_stage(self.stage_detail[stage])
        return stage_config["max_iterations"]


    def _build_runner(
        self,
        component: Component,
        max_iterations: int,
        transition: list[dict] | None = None,
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
                transition=transition,
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
                transition=transition,
                **runner_config
            )

        

    def test(self, num_episodes: int) -> None:

        if hasattr(self, "runner"):
            if self.continue_training:
                logger.warning("Model is not trained completely.")
            self.runner.test(num_episodes=num_episodes)
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
        self.runner.test(num_episodes=num_episodes)


    def play(
        self,
        num_steps: int,
        formats: VideoFormat | VideoFormats,
        num_plays: int,
    ) -> None:

        if hasattr(self, "runner"):
            if self.continue_training:
                logger.warning("Model is not trained completely.")
            self.runner.play(
                num_steps=num_steps,
                formats=formats,
                num_plays=num_plays,
            )
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
        self.runner.play(
            num_steps=num_steps,
            formats=formats,
            num_plays=num_plays,
        )


    def _prepare_evaluation_stage(self) -> Path | None:

        if hasattr(self, "runner"):
            return None

        selection = self.resume_selection
        if selection is None:
            return None

        self.current_stage = selection.stage_index
        return selection.checkpoint


    def _load_evaluation_checkpoint(
        self,
    ) -> None:
        
        if not hasattr(self, "runner"):
            raise RuntimeError("Runner was not built for checkpoint loading.")
        self.runner.load(mode=RestoreMode.EVALUATE)
        logger.info("Restored evaluation checkpoint (optimizer excluded).")


    def save(self) -> Path:
        
        if not hasattr(self, "runner"):
            raise RuntimeError("Cannot save before a runner has been built.")
        checkpoint_path = self.runner.save()
        logger.info(
            "Saved checkpoint to %s.",
            checkpoint_path.as_posix()
        )
        return checkpoint_path


    def close(self) -> None:

        if hasattr(self, "runner"):
            self.runner.close()
            del self.runner
