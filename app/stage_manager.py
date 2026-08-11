import warnings
from pathlib import Path

from utils.component import create_component, Component
from utils.config import load_yaml
from app.utils.context import RuntimeContext
from runners.callbacks.stage import StageCallback
from runners.registry import RUNNER_TYPE_MAP


class StageManager:

    def __init__(
        self,
        component: dict,
        context: RuntimeContext,
        stage_detail: list[dict],
    ) -> None:
        self.component = component
        self.context = context
        self.stage_detail = stage_detail
        self.runner = None
        self.current_stage = None
        self.max_stage = None
        self._setup()


    @property
    def continue_training(self) -> bool:

        if self.current_stage <= self.max_stage:
            return True 
        
        return False


    def _setup(self) -> None:

        self.current_stage = 0
        self.max_stage = len(self.stage_detail) - 1

        for name, detail in self.component.items():
            if len(detail) - 1 > self.max_stage:
                raise ValueError(
                    f"{name} config size does not match stages."
                )


    def _build_runner(
        self,
        component: Component,
        stage_callback: StageCallback,
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
            if not isinstance(self.runner, runner_type):
                self.runner = runner_type(
                    context=self.context
                )
            self.runner.config_update(
                component=component,
                **runner_config
            )

            stage_callback.set_runner(self.runner)
            self.runner.stage_update(stage_callback)


    def train(self) -> None:
        
        while self.continue_training:
            stage_complete = self.train_current()

            if stage_complete:
                self.current_stage += 1
            else:
                warnings.warn(
                    f"Stage {self.current_stage} timeout."
                )
                break


    def train_current(self) -> bool:

        current_component = self._get_current_component()
        current_stage_callback = self._build_current_callback()

        self._build_runner(
            component=current_component,
            stage_callback=current_stage_callback,
        )
        self.runner.train()

        return current_stage_callback.stop_training


    def _get_current_component(self) -> Component:

        current_component = dict()

        for name, detail in self.component.items():

            stage = self.current_stage
            if stage <= len(detail) - 1:

                if stage > 0 and detail[stage] == detail[stage - 1]:
                    continue
            
                current_component[name] = detail[stage]

        return create_component(current_component)


    def _build_current_callback(self) -> StageCallback:
        condition_dict = self.stage_detail[self.current_stage]
        return StageCallback(
            condition=condition_dict
        )
        

    def test(self) -> None:
    
        if self.continue_training:
            warnings.warn("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component
        )
        self.runner.test()


    def play(self) -> None:

        if self.stage_manager.continue_training:
            warnings.warn("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            component=current_component
        )
        self.runner.play()


    def close(self) -> None:

        self.runner.close()
