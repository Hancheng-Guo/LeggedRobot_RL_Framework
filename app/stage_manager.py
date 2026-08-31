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
    STOPPED_BY_CALLBACK = auto()
    MAX_ITERATIONS_REACHED = auto()


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
        
        while self.continue_training:
            train_result = self._train_current()

            if train_result is StageTrainResult.STAGE_COMPLETED:
                self.current_stage += 1
                continue

            if train_result is StageTrainResult.STOPPED_BY_CALLBACK:
                for callback in self.runner.stop_callback:
                    logger.warning(
                        f"Training was stopped by callback "
                        f"{type(callback).__name__!r} during stage "
                        f"{self.current_stage}."
                    )
            else:
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


    def _train_current(self) -> StageTrainResult:

        if not self.continue_training:
            raise RuntimeError("All stages have already been completed.")

        current_component = self._get_current_component()
        current_stage_callback = self._build_current_stage_callback()

        self._build_runner(
            component=current_component,
            stage_callback=current_stage_callback,
            max_iterations=self._get_current_max_iterations(),
        )
        self.runner.train()

        if current_stage_callback.stop_training:
            return StageTrainResult.STAGE_COMPLETED
        if self.runner.stop_callback:
            return StageTrainResult.STOPPED_BY_CALLBACK
        return StageTrainResult.MAX_ITERATIONS_REACHED


    def _get_current_component(self) -> Component:

        current_component = dict()
        stage = self._get_effective_stage()

        for name, detail in self.component.items():

            if stage <= len(detail) - 1:

                if stage > 0 and detail[stage] == detail[stage - 1]:
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
    ) -> None:

        if component.runner is None:
            if self.runner is None:
                raise RuntimeError(
                    f"runner instance is required."
                )
            self.runner.config_update(
                component=component,
                max_iterations=max_iterations,
            )

        else:
            runner_type_name = component.runner.type
            if runner_type_name not in RUNNER_TYPE_MAP:
                raise ValueError(
                    f"Invalid runner type: {runner_type_name!r}. "
                )

            runner_type = RUNNER_TYPE_MAP[runner_type_name]
            runner_config = load_yaml(component.runner.config)

            if(
                not hasattr(self, "runner")
                or not isinstance(self.runner, runner_type)
            ):
                self.runner = runner_type(
                    context=self.context
                )
            self.runner.config_update(
                component=component,
                max_iterations=max_iterations,
                **runner_config
            )

        if stage_callback is not None:
            stage_callback.set_runner(self.runner)
        self.runner.stage_update(stage_callback)
        

    def test(self, *args, **kwargs) -> None:
    
        if self.continue_training:
            logger.warning("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component,
            max_iterations=self._get_current_max_iterations(),
        )
        self.runner.test(*args, **kwargs)


    def play(self, *args, **kwargs) -> None:

        if self.continue_training:
            logger.warning("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component,
            max_iterations=self._get_current_max_iterations(),
        )
        self.runner.play(*args, **kwargs)


    def save(self) -> None:
        raise NotImplementedError


    def close(self) -> None:

        if self.runner is not None:
            self.runner.close()
