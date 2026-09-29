"""Load an Isaac manual-test configuration without adding a formal task."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.application_entry import ApplicationEntry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_APP_NAME = "unitree_go1_isaac_cuda_headless_train_test"
TEST_CONFIG = Path(__file__).resolve().parent / "configs" / f"{TEST_APP_NAME}.yaml"


class _ManualIsaacApplication(ApplicationEntry):
    def __init__(self, config_root: Path) -> None:
        self._manual_config_root = config_root
        super().__init__(TEST_APP_NAME)

    def _get_runtime_dir(self, skip_check: bool) -> tuple[Path, Path]:
        if not skip_check:
            raise ValueError("Manual Isaac test only starts a new run.")
        timestamp = self.train_time.strftime("%Y-%m-%d_%H-%M-%S")
        return self._manual_config_root, Path("checkpoints") / f"{TEST_APP_NAME}_{timestamp}"


@contextmanager
def manual_isaac_application() -> Iterator[ApplicationEntry]:
    """Stage the fixture alongside component YAML files for one test run."""
    with tempfile.TemporaryDirectory(prefix="isaac_manual_config_") as directory:
        config_root = Path(directory)
        shutil.copytree(PROJECT_ROOT / "configs", config_root / "configs")
        shutil.copy2(TEST_CONFIG, config_root / "configs" / TEST_CONFIG.name)
        with _ManualIsaacApplication(config_root) as app:
            yield app
