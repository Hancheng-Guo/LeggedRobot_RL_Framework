import warnings
from pathlib import Path
from datetime import datetime

from app.stage_manager import StageManager
from app.utils.context import create_runtime_context, RuntimeContext
from utils.config import load_yaml


class ApplicationEntry:

    def __init__(
        self,
        app_name: str,
        train_time: str | None = None,
    ) -> None:
        
        self.app_name: str
        self.train_time: datetime | None
        self.load_dir: Path
        self.config: dict
        self.context: RuntimeContext
        self.stage_manager: StageManager

        self._setup(app_name, train_time)

    
    def _setup(
        self,
        app_name: str,
        train_time: str | None = None,
    ) -> None:

        self.app_name = app_name

        if train_time is None:
            self.train_time = None

        else:
            try:
                self.train_time = datetime.strptime(
                    train_time, "%Y-%m-%d_%H-%M-%S"
                )

            except ValueError as e:
                warnings.warn(
                    f"{str(e)}, 'train_time' field will be remove."
                )
                self.train_time = None
            
        self.load_dir, self.save_dir = self._get_runtime_dir()

        self.config = load_yaml(
            self.load_dir / "configs" / f"{self.app_name}.yaml"
        )

        runtime_config = self.config.get("runtime")
        if not isinstance(runtime_config, dict):
            raise ValueError("runtime config is not a instance of 'dict'")
        self.context = create_runtime_context(
            runtime_config=runtime_config,
            load_dir=self.load_dir,
            save_dir=self.save_dir,
        )

        component = self.config.get("component")
        if not isinstance(component, dict):
            raise ValueError("component config is not a instance of 'dict'")
        stage_detail = self.config.get("stage")
        if not isinstance(stage_detail, list):
            raise ValueError("stage_detail config is not a instance of 'list'")
        self.stage_manager = StageManager(
            component=component,
            context=self.context,
            stage_detail=stage_detail,
            load_dir=self.load_dir,
        )


    def _get_runtime_dir(self) -> tuple[Path, Path]:

        if self.train_time is not None:

            runtime_dir = Path(
                f"./checkpoints"
                f"/{self.app_name}_{
                    self.train_time.strftime("%Y-%m-%d_%H-%M-%S")
                }"
            )
            base_config_file = runtime_dir / "configs" / f"{self.app_name}.yaml"

            if base_config_file.is_file():
                return runtime_dir, runtime_dir     # load_dir, save_dir

            warnings.warn(f"checkpoint directory '{runtime_dir}' is incomplete.")

        return Path("."), runtime_dir   # load_dir, save_dir


    def train(self) -> None:

        train_time = datetime.now()
        self.stage_manager.train()
        self.train_time = train_time


    def test(self, *args, **kwargs) -> None:

        self.stage_manager.test(*args, **kwargs)


    def play(self, *args, **kwargs) -> None:

        self.stage_manager.play(*args, **kwargs)


    def save(self) -> None:

        self.stage_manager.save()


    def close(self) -> None:

        self.stage_manager.close()
