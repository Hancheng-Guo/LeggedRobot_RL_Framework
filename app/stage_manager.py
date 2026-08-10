import warnings
from pathlib import Path


from utils.component import create_component, Component
from app.utils.context import RuntimeContext
from runners.registry import RUNNER_TYPE_MAP


class StageManager:

    def __init__(
        self,
        component: dict,
        context: RuntimeContext,
    ) -> None:
        self.component = component
        self.context = context
        self.runner = None
        self.current_stage = None
        self.max_stage = None
        self._setup()


    @property
    def continue_training(self) -> bool:

        if self.current_stage >= (self.max_stage + 1):
            return True 
        
        return False


    def _setup(self) -> None:

        self.current_stage = 0
        max_stage = 0

        for component_detail in self.component.values():
            if len(component_detail) - 1 > max_stage:
                max_stage = len(component_detail) - 1

        self.max_stage = max_stage


    def _build_runner(
        self,
        current_component: Component
    ) -> None:
        
        runner_type_name = current_component.runner.type
        if runner_type_name not in RUNNER_TYPE_MAP:
            raise ValueError(
                f"Invalid runner type: {runner_type_name!r}. "
            )

        runner_type = RUNNER_TYPE_MAP[runner_type_name]
        if isinstance(self.runner, runner_type):
            runner_config = current_component.runner.config
            self.runner.config_update(
                component=current_component,
                **runner_config
            )
        else:
            self.runner = runner_type(
                component=current_component,
                **runner_config
            )


    def train(self) -> None:
        
        while self.continue_training:
            self.train_current()
            self.current_stage += 1


    def train_current(self) -> None:

        current_component = self._get_current_component()
        self._build_runner(
            current_component=current_component
        )
        self.runner.train()


    def _get_current_component(self) -> Component:

        current_component = dict()

        for component_name, component_detail in self.component.items():
            if self.current_stage > len(component_detail) - 1:
                current_index = len(component_detail) - 1
            else:
                current_index = self.current_stage
            current_component[component_name] = component_detail[current_index]

        return create_component(current_component)


    def test(self) -> None:
    
        if self.continue_training:
            warnings.warn("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            current_component=current_component
        )
        self.runner.test()


    def play(self) -> None:

        if self.stage_manager.continue_training:
            warnings.warn("Model is not trained completely.")
        current_component = self._get_current_component()
        self._build_runner(
            current_component=current_component
        )
        self.runner.play()


    def close(self) -> None:

        self.runner.close()
