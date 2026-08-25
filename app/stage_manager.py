import warnings
from enum import Enum, auto
from pathlib import Path

from app.utils.context import RuntimeContext
from runners.callbacks.stage import StageCallback
from runners.base import BaseRunner
from runners.registry import RUNNER_TYPE_MAP
from utils.component import create_component, Component
from utils.config import load_yaml


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

        for name, detail in self.component.items():
            if len(detail) - 1 > self.max_stage:
                raise ValueError(
                    f"{name} config size does not match stages."
                )


    def train(self) -> None:
        
        while self.continue_training:
            train_result = self._train_current()

            if train_result is StageTrainResult.STAGE_COMPLETED:
                self.current_stage += 1
                continue

            if train_result is StageTrainResult.STOPPED_BY_CALLBACK:
                for callback in self.runner.stop_callback:
                    warnings.warn(
                        f"Training was stopped by callback "
                        f"{type(callback).__name__!r} during stage "
                        f"{self.current_stage}."
                    )
            else:
                warnings.warn(f"Stage {self.current_stage} timeout.")
            break


    @property
    def continue_training(self) -> bool:

        if self.current_stage <= self.max_stage:
            return True 
        
        return False


    def _train_current(self) -> StageTrainResult:

        current_component = self._get_current_component()
        current_stage_callback = self._build_current_stage_callback()

        self._build_runner(
            component=current_component,
            stage_callback=current_stage_callback,
        )
        self.runner.train()

        if current_stage_callback.stop_training:
            return StageTrainResult.STAGE_COMPLETED
        if self.runner.stop_callback:
            return StageTrainResult.STOPPED_BY_CALLBACK
        return StageTrainResult.MAX_ITERATIONS_REACHED


    def _get_current_component(self) -> Component:

        current_component = dict()

        for name, detail in self.component.items():

            stage = self.current_stage
            if stage <= len(detail) - 1:

                if stage > 0 and detail[stage] == detail[stage - 1]:
                    continue
            
                current_component[name] = detail[stage]

        return create_component(current_component, self.load_dir)


    def _build_current_stage_callback(self) -> StageCallback:
        condition_dict = self.stage_detail[self.current_stage]
        return StageCallback(
            condition=condition_dict
        )


    def _build_runner(
        self,
        component: Component,
        stage_callback: StageCallback | None = None,
    ) -> None:

        if component.runner is None:
            if self.runner is None:
                raise RuntimeError(
                    f"runner instance is required."
                )
            self.runner.config_update(component=component)

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
                **runner_config
            )

        if stage_callback is not None:
            stage_callback.set_runner(self.runner)
        self.runner.stage_update(stage_callback)
        

    def test(self, *args, **kwargs) -> None:
    
        if self.continue_training:
            warnings.warn("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component
        )
        self.runner.test(*args, **kwargs)


    def play(self, *args, **kwargs) -> None:

        if self.continue_training:
            warnings.warn("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component
        )
        self.runner.play(*args, **kwargs)


    def save(self) -> None:
        raise NotImplementedError


    def close(self) -> None:

        if self.runner is not None:
            self.runner.close()
