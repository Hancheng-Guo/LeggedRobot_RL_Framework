import sys
from pathlib import Path
from typing import Any

import pytest

from tools import launch_tensorboard as launcher
from tools.launch_tensorboard import select_logdir
from utils import select_run_dir


def test_select_logdir_uses_latest_timestamp_not_directory_order(
    tmp_path: Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    older = checkpoints / "z_robot_2026-09-30_23-59-59" / "tensorboard"
    latest = checkpoints / "a_robot_2026-10-01_00-00-00" / "tensorboard"
    older.mkdir(parents=True)
    latest.mkdir(parents=True)
    (checkpoints / "undated" / "tensorboard").mkdir(parents=True)

    assert select_run_dir(checkpoints) == latest.parent.resolve()
    assert select_run_dir(checkpoints, older.parent.name) == older.parent.resolve()
    assert select_run_dir(checkpoints, str(older.parent)) == older.parent.resolve()
    assert select_logdir(checkpoints) == latest.resolve()
    assert select_logdir(checkpoints, older.parent.name) == older.resolve()
    assert select_logdir(checkpoints, str(older)) == older.resolve()


def test_select_logdir_reports_missing_latest_logdir(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "robot_2026-10-01_00-00-00").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="TensorBoard log directory"):
        select_logdir(checkpoints)

    assert select_run_dir(checkpoints) == (
        checkpoints / "robot_2026-10-01_00-00-00"
    ).resolve()

    with pytest.raises(FileNotFoundError, match="Run directory"):
        select_run_dir(checkpoints, "missing_run")


def test_main_resolves_empty_input_before_starting_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logdir = tmp_path / "checkpoints" / "run_2026-10-02_00-00-00" / "tensorboard"
    logdir.mkdir(parents=True)
    commands: list[list[str]] = []

    class FakeProcess:
        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        return FakeProcess()

    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sys, "argv", ["launch_tensorboard", ""])

    assert launcher.main() == 0
    assert commands == [[
        sys.executable,
        "-m",
        "tools.launch_tensorboard",
        "--serve",
        str(logdir.resolve()),
    ]]
