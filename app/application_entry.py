from pathlib import Path
from datetime import datetime

from utils.config import load_yaml, get_yaml_value
from app.utils.context import create_runtime_context
from runners.registry import RUNNER_TYPE_MAP


class ApplicationEntry:

    def __init__(
            self,
            app_name: str = None,
    ) -> None:
        self.app_name = app_name
        self.config = None
        self.context = None
        self.runner = None
        self.latest_train_time = None
        self._setup()

    
    def _setup(self) -> None:

        self.config = load_yaml(
            Path(get_yaml_value("configs/base.yaml", "config_dir"))
            / self.app_name
        )

        runtime_config = self.config.get("runtime")
        self.context = create_runtime_context(
            runtime_config=runtime_config
        )

        self._build_runner()


    def _build_runner(self) -> None:

        runner_config = self.config.get("runner")
        
        runner_type_name = runner_config.get("runner_type")
        if runner_type_name not in RUNNER_TYPE_MAP:
            raise ValueError(
                f"Invalid runner type: {runner_type_name!r}. "
            )
    
        runner_type = RUNNER_TYPE_MAP[runner_type_name]
        self.runner = runner_type(
            component_detail=self.config.get("component"),
            **runner_config,
        )


    def train(self) -> None:
        latest_train_time = datetime.now()
        self.runner.train()
        self.latest_train_time = latest_train_time


    def test(self) -> None:
        if self.latest_train_time is None:
            raise RuntimeError("No trained model is available.")
        self.runner.test()


    def play(self) -> None:
        if self.latest_train_time is None:
            raise RuntimeError("No trained model is available.")
        self.runner.play()


    def close(self) -> None:
        self.runner.close()
        