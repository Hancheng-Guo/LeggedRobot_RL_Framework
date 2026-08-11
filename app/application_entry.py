from pathlib import Path
from datetime import datetime

from app.stage_manager import StageManager
from utils.config import load_yaml, get_yaml_value
from app.utils.context import create_runtime_context


class ApplicationEntry:

    def __init__(
            self,
            app_name: str = None,
    ) -> None:
        self.app_name = app_name
        self.config = None
        self.context = None
        self.stage_manager = None
        self.train_time = None
        self._setup()

    
    def _setup(self) -> None:

        self.config = load_yaml(
            Path(get_yaml_value("configs/base.yaml", "config_dir"))
            / self.app_name
        )

        self.context = create_runtime_context(
            runtime_config=self.config.get("runtime")
        )

        self.stage_manager = StageManager(
            component=self.config.get("component"),
            context=self.context,
            stage_detail=self.config.get("stage"),
        )


    def train(self) -> None:

        train_time = datetime.now()
        self.stage_manager.train()
        self.train_time = train_time


    def test(self) -> None:

        self.stage_manager.test()


    def play(self) -> None:

        self.stage_manager.play()


    def close(self) -> None:

        self.stage_manager.close()
        