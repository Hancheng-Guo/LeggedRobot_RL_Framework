import warnings
import shutil
from pathlib import Path
from datetime import datetime

from app.stage_manager import StageManager
from app.utils.context import create_runtime_context, RuntimeContext
from utils.config import load_yaml
from utils.logging import configure_logging, get_logger, LoggingSession


logger = get_logger(__name__)


class ApplicationEntry:

    def __init__(
        self,
        app_name: str,
        train_time: str | None = None,
        device: str | None = None,
    ) -> None:
        
        self.app_name: str
        self.train_time: datetime
        self.load_dir: Path
        self.config: dict
        self.context: RuntimeContext
        self.stage_manager: StageManager
        self.logging_session: LoggingSession
        self._closed = False

        self._setup(app_name, train_time, device)

    
    def _setup(
        self,
        app_name: str,
        train_time: str | None = None,
        device: str | None = None,
    ) -> None:

        self.app_name = app_name

        if train_time is None:
            self.train_time = datetime.now()
            skip_check = True

        else:

            try:
                self.train_time = datetime.strptime(
                    train_time, "%Y-%m-%d_%H-%M-%S"
                )
                skip_check = False

            except ValueError:
                warnings.warn(
                    f"Invalid train_time '{train_time}', using the current time instead."
                )
                self.train_time = datetime.now()
                skip_check = True

        self.load_dir, self.save_dir = self._get_runtime_dir(
            skip_check=skip_check
        )

        self.config = load_yaml(
            self.load_dir / "configs" / f"{self.app_name}.yaml"
        )

        logging_config = self.config.get("logging", {})
        if not isinstance(logging_config, dict):
            raise TypeError("logging config must be a mapping.")
        logging_config = dict(logging_config)
        file_name = logging_config.pop("file_name", "training.log")
        console = logging_config.pop("console", True)
        if logging_config:
            raise ValueError(
                "Unsupported logging configuration: "
                f"{sorted(logging_config)}."
            )
        if not isinstance(file_name, str) or not file_name:
            raise ValueError("logging.file_name must be a non-empty string.")
        if not isinstance(console, bool):
            raise TypeError("logging.console must be a boolean.")
        self.logging_session = configure_logging(
            log_file=self.save_dir / "logs" / file_name,
            console=console,
        )

        try:
            configured_runtime = self.config.get("runtime")
            if not isinstance(configured_runtime, dict):
                raise ValueError("runtime config is not a instance of 'dict'")
            runtime_config = dict(configured_runtime)
            if device is not None:
                configured_device = runtime_config.get("device")
                runtime_config["device"] = device
                logger.warning(
                    "Runtime device overridden for this run: "
                    f"{configured_device!r} -> {device!r}. "
                    "The saved configuration is unchanged."
                )
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
        except Exception:
            logger.exception("Application setup failed.")
            self.logging_session.close()
            raise


    def _get_runtime_dir(
        self,
        skip_check: bool,
    ) -> tuple[Path, Path]:

        runtime_dir = Path(
            f"./checkpoints"
            f"/{self.app_name}_{
                self.train_time.strftime("%Y-%m-%d_%H-%M-%S")
            }"
        )

        if skip_check:
            return Path("."), runtime_dir       # load_dir, save_dir

        base_config_file = runtime_dir / "configs" / f"{self.app_name}.yaml"

        if base_config_file.is_file():
            return runtime_dir, runtime_dir     # load_dir, save_dir

        raise FileNotFoundError(
            f"Historical training config is missing: {base_config_file}."
        )


    def train(self) -> None:
        self._ensure_open()
        self._save_configs()
        self.stage_manager.train()


    def test(self, *args, **kwargs) -> None:
        self._ensure_open()
        self.stage_manager.test(*args, **kwargs)


    def play(self, *args, **kwargs) -> None:
        self._ensure_open()
        self.stage_manager.play(*args, **kwargs)


    def save(self) -> Path:
        self._ensure_open()
        self._save_configs()
        return self.stage_manager.save()


    def _save_configs(self) -> None:
        config_source = self.load_dir / "configs"
        config_destination = self.save_dir / "configs"
        if (
            config_source.is_dir()
            and config_source.resolve() != config_destination.resolve()
        ):
            shutil.copytree(
                config_source,
                config_destination,
                dirs_exist_ok=True,
            )


    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        try:
            self.stage_manager.close()
        finally:
            logging_session = getattr(self, "logging_session", None)
            if logging_session is not None:
                logging_session.close()
            self._closed = True


    def _ensure_open(self) -> None:
        if getattr(self, "_closed", False):
            raise RuntimeError("ApplicationEntry is closed.")


    def __enter__(self) -> "ApplicationEntry":
        self._ensure_open()
        return self


    def __exit__(self, *args) -> None:
        self.close()
