import sys
from types import ModuleType

import pytest

from tools import resume_training


@pytest.fixture
def resume_workspace(tmp_path, monkeypatch):
    run_dir = tmp_path / "checkpoints" / "robot_2026-10-01_12-30-05"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(resume_training, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    return run_dir


def install_fake_application(monkeypatch, calls, *, fail_at=None):
    class FakeApplication:
        def __init__(self, app_name, **kwargs):
            calls.append((app_name, kwargs))

        def train(self):
            calls.append("train")
            if fail_at == "train":
                raise RuntimeError("train failed")

        def save(self):
            calls.append("save")
            if fail_at == "save":
                raise RuntimeError("save failed")

        def close(self):
            calls.append("close")

    module = ModuleType("app")
    module.ApplicationEntry = FakeApplication  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app", module)


@pytest.mark.parametrize("selected", ["", "robot_2026-10-01_12-30-05"])
def test_resume_task_uses_archived_run_and_default_options(resume_workspace, monkeypatch, selected):
    calls = []
    install_fake_application(monkeypatch, calls)
    resume_training.main([selected, "--checkpoint="])
    assert calls == [
        ("robot", {"train_time": "2026-10-01_12-30-05", "resume_mode": "fork", "checkpoint": None}),
        "train", "save", "close",
    ]


@pytest.mark.parametrize("mode", ["fork", "inplace"])
def test_resume_task_forwards_checkpoint_and_output_mode(resume_workspace, monkeypatch, mode):
    calls = []
    install_fake_application(monkeypatch, calls)
    checkpoint = "stage_000/checkpoint with spaces.pt"
    resume_training.main([
        str(resume_workspace), f"--checkpoint={checkpoint}", "--resume-mode", mode,
    ])
    assert calls[0] == ("robot", {
        "train_time": "2026-10-01_12-30-05", "resume_mode": mode, "checkpoint": checkpoint,
    })
    assert calls[1:] == ["train", "save", "close"]


@pytest.mark.parametrize("fail_at", ["train", "save"])
def test_resume_task_closes_application_after_error(resume_workspace, monkeypatch, fail_at):
    calls = []
    install_fake_application(monkeypatch, calls, fail_at=fail_at)
    with pytest.raises(RuntimeError, match=f"{fail_at} failed"):
        resume_training.main([resume_workspace.name])
    assert calls[-1] == "close"
    if fail_at == "train":
        assert "save" not in calls


def test_resume_task_rejects_run_outside_project_checkpoints(resume_workspace, tmp_path, monkeypatch):
    outside = tmp_path / "robot_2026-10-02_12-30-05"
    outside.mkdir()
    calls = []
    install_fake_application(monkeypatch, calls)
    with pytest.raises(SystemExit):
        resume_training.main([str(outside)])
    assert not calls
