from __future__ import annotations

import yaml
import shutil
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime
from types import TracebackType
from typing import TYPE_CHECKING


from app.stage_manager import StageManager
from app.utils.run_workspace import (
    select_checkpoint,
    inspect_checkpoint_metadata,
    CheckpointSelection,
)
from app.utils.context import create_runtime_context
from utils.runtime import RuntimeContext
from utils.component import create_component_info
from utils.config import load_yaml
from utils.logging import configure_logging, get_logger, LoggingSession
from utils.component import COMPONENT_CONFIG_DIR_MAP

if TYPE_CHECKING:
    from runners.utils.frames import VideoFormat, VideoFormats


logger = get_logger(__name__)


@dataclass(frozen=True)
class _LoggingConfig:
    file_name: str
    console: bool


class ApplicationEntry:

    def __init__(
        self,
        app_name: str,
        train_time: str | None = None,
        device: str | None = None,
        *,
        resume_mode: str = "fork",
        checkpoint: str | Path | None = None,
    ) -> None:
        
        self.app_name: str
        self.train_time: datetime
        self.load_dir: Path
        self.config: dict
        self.context: RuntimeContext
        self.stage_manager: StageManager
        self._resume_selection: CheckpointSelection | None
        self._logging_config: _LoggingConfig
        self.logging_session: LoggingSession
        self._closed = False
        self._configs_saved = False
        self._fork_prepared = False

        self._setup(app_name, train_time, device, resume_mode, checkpoint)

    
    def _setup(
        self,
        app_name: str,
        train_time: str | None,
        device: str | None,
        resume_mode: str,
        checkpoint: str | Path | None,
    ) -> None:

        if resume_mode not in {"fork", "inplace"}:
            raise ValueError("resume_mode must be 'fork' or 'inplace'.")
        if checkpoint is not None and train_time is None:
            raise ValueError("checkpoint requires train_time.")
        
        self.app_name = app_name
        self.resume_mode = resume_mode
        self._historical = train_time is not None

        if train_time is None:
            self.train_time = datetime.now()
            skip_check = True

        else:

            try:
                self.train_time = datetime.strptime(train_time, "%Y-%m-%d_%H-%M-%S")
                if self.train_time.strftime("%Y-%m-%d_%H-%M-%S") != train_time:
                    raise ValueError
                skip_check = False
            except (ValueError, TypeError):
                raise ValueError(f"Invalid train_time {train_time!r}; expected YYYY-MM-DD_HH-MM-SS.") from None

        self.load_dir, self.save_dir = self._get_runtime_dir(
            skip_check=skip_check
        )
        self.config = load_yaml(
            self.load_dir / "configs" / f"{self.app_name}.yaml"
        )

        self._logging_config, self.logging_session = self._setup_logging()
        logger.info(
            "Application %s initialized.\n"
            "    Config : %s\n"
            "    Output : %s\n"
            "    Log    : %s",
            self.app_name,
            (self.load_dir / "configs" / f"{self.app_name}.yaml").as_posix(),
            self.save_dir.as_posix(),
            (self.save_dir / "logs" / self._logging_config.file_name).as_posix(),
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
            self._resume_selection = (
                select_checkpoint(
                    run_dir=self.load_dir,
                    stage_count=len(stage_detail),
                    selected=checkpoint,
                )
                if self._historical else None
            )
            self.stage_manager = StageManager(
                component=component,
                context=self.context,
                stage_detail=stage_detail,
                load_dir=self.load_dir,
                resume_selection=self._resume_selection,
            )
        except Exception:
            logger.exception("Application setup failed.")
            self.logging_session.close()
            raise


    def _setup_logging(self) -> tuple[_LoggingConfig, LoggingSession]:

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
        
        config = _LoggingConfig(
            file_name=file_name,
            console=console
        )
        # Delay file logging until train() creates the fork directory.
        session = (
            LoggingSession(logger=get_logger())
            if self._historical and self.resume_mode == "fork"
            else configure_logging(
                log_file=self.save_dir / "logs" / config.file_name,
                console=config.console,
            )
        )
        return config, session


    def _get_runtime_dir(
        self,
        skip_check: bool,
    ) -> tuple[Path, Path]:

        runtime_dir = Path(
            f"./checkpoints"
            f"/{self.app_name}_{self.train_time.strftime('%Y-%m-%d_%H-%M-%S')}"
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
        if (
            self._historical
            and self.resume_mode == "fork"
            and not self._fork_prepared
        ):
            self._prepare_fork()
        if self._save_configs():
            logger.info(
                "Training configuration saved to %s.",
                (self.save_dir / "configs").as_posix()
            )
        self.stage_manager.train()


    def _prepare_fork(self) -> None:

        self.stage_manager.close()

        selection = self._resume_selection
        if selection is None:
            raise RuntimeError("Fork requires a selected checkpoint.")
        source, completed = selection.checkpoint, selection.stage_completed

        now = datetime.now()
        run_time = now.replace(microsecond=0)
        destination = (
            Path("checkpoints")
            / f"{self.app_name}_{run_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        if destination.exists():
            raise FileExistsError(f"Fork run directory already exists: {destination}.")
        preparing_dir = destination.with_name(f".{destination.name}.preparing")
        preparing_dir.mkdir(parents=True, exist_ok=False)

        try:
            checkpoint_dir = preparing_dir / "checkpoints"
            checkpoint_dir.mkdir()

            archived = checkpoint_dir / "resume.pt"
            shutil.copy2(source, archived)

            metadata = inspect_checkpoint_metadata(archived)
            stage_index = metadata.stage_index
            if stage_index is None or stage_index != selection.stage_index:
                raise ValueError(
                    "Copied checkpoint stage differs from the selected source."
                )
            
            manifest = {
                "version": 1,
                "mode": "fork",
                "status": "preparing",
                "source_run": str(self.load_dir.resolve()),
                "source_checkpoint": str(source.resolve()),
                "resume_checkpoint": "checkpoints/resume.pt",
                "stage_completed": metadata.stage_completed,
                "stage_index": stage_index,
                "current_iteration": metadata.current_iteration,
                "device": str(self.context.device),
                "seed": self.context.seed,
                "created_at": now.isoformat(),
                "run_time": run_time.strftime("%Y-%m-%d_%H-%M-%S"),
            }
            manifest_path = preparing_dir / "resume_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            config_copy = self._archive_configs(preparing_dir)
            manifest["status"] = "ready"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if destination.exists():
                raise FileExistsError(f"Fork run directory already exists: {destination}.")
            preparing_dir.rename(destination)

            logging_session = configure_logging(
                destination / "logs" / self._logging_config.file_name,
                self._logging_config.console,
            )
            self.logging_session.close()
            self.save_dir = destination
            self.logging_session = logging_session

            self.context = replace(self.context, load_dir=destination, save_dir=destination)
            self.load_dir = destination
            self.train_time = run_time
            self.config = config_copy
            self._resume_selection = CheckpointSelection(
                destination / "checkpoints" / "resume.pt",
                stage_index,
                completed
            )

            self.stage_manager.use_fork_workspace(
                component=self.config["component"],
                context=self.context,
                load_dir=self.load_dir,
                resume_selection=self._resume_selection,
            )
            
            self._configs_saved = True
            self._fork_prepared = True
            logger.info("Fork prepared from %s; output: %s.", source, destination)

        except Exception:
            # Failed preparation remains outside the public run namespace.
            logger.exception("Failed to prepare fork from %s.", source)
            raise


    def _archive_configs(
        self,
        destination: Path
    ) -> dict:
        
        source_root = (self.load_dir / "configs").resolve()
        dest_root = destination / "configs"
        dest_root.mkdir()
        
        config_copy = deepcopy(self.config)
        for name, entries in config_copy["component"].items():
            for entry in entries:
                info = create_component_info({name: entry}, name, self.load_dir)
                if info is None:
                    continue
                source = info.config.resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"Component configuration is missing: {source}.")
                # Give every distinct source a stable name in the default
                # component directory, including external config paths.
                default_dir = source_root / COMPONENT_CONFIG_DIR_MAP[name]
                if source.parent == default_dir:
                    target_name = source.name
                else:
                    prefix = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
                    target_name = f"external_{prefix}.yaml"
                target = dest_root / COMPONENT_CONFIG_DIR_MAP[name] / target_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                entry.pop("config_path", None)
                entry.pop("config_dir", None)
                entry["config"] = target_name

        (dest_root / f"{self.app_name}.yaml").write_text(
            yaml.safe_dump(config_copy, sort_keys=False),
            encoding="utf-8"
        )
        return config_copy


    def test(
        self,
        num_episodes: int = 1000
    ) -> None:
        self._ensure_open()
        self.stage_manager.test(num_episodes=num_episodes)


    def play(
        self,
        num_steps: int = 500,
        formats: VideoFormat | VideoFormats = "gif",
        num_plays: int = 3,
    ) -> None:
        self._ensure_open()
        self.stage_manager.play(
            num_steps=num_steps,
            formats=formats,
            num_plays=num_plays,
        )


    def save(self) -> Path:
        self._ensure_open()
        return self.stage_manager.save()


    def _save_configs(self) -> bool:

        if self._configs_saved:
            return False
        config_source = self.load_dir / "configs"
        config_destination = self.save_dir / "configs"

        if (
            config_source.is_dir()
            and config_source.resolve() != config_destination.resolve()
        ):
            source_root = config_source.resolve()
            config_paths = {config_source / f"{self.app_name}.yaml"}
            component_config = self.config["component"]
            for name, entries in component_config.items():
                for entry in entries:
                    component_info = create_component_info(
                        {name: entry}, name, self.load_dir
                    )
                    if component_info is not None:
                        config_paths.add(component_info.config)

            for config_path in config_paths:
                source_path = config_path.resolve()
                try:
                    relative_path = source_path.relative_to(source_root)
                except ValueError:
                    logger.warning(
                        "External component config is not archived: %s.",
                        source_path.as_posix(),
                    )
                    continue

                destination_path = config_destination / relative_path
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)

            self._configs_saved = True
            return True
        
        return False


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


    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is not None and exception is not None:
            logger.error(
                "Application execution failed.",
                exc_info=(exception_type, exception, traceback),
            )
            self.logging_session.flush()
        self.close()
