import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
import yaml

from app.application_entry import ApplicationEntry


STAMP = "2026-01-02_03-04-05"


def source_run(tmp_path: Path, app_name: str = "app") -> Path:
    run = tmp_path / "checkpoints" / f"{app_name}_{STAMP}"
    (run / "configs" / "runners").mkdir(parents=True)
    (run / "configs" / "runners" / "runner.yaml").write_text("rollout_length: 8\n", encoding="utf-8")
    config = {
        "runtime": {"device": "cpu", "dtype": "float32", "num_threads": 1,
                    "seed": 5, "deterministic_ops": False},
        "logging": {"console": False},
        "stage": [{"one": {"max_iterations": 10, "transition": [True]}}],
        "component": {"runner": [{"type": "on_policy", "config": "runner"}]},
    }
    (run / "configs" / f"{app_name}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = run / "checkpoints" / "stage_000" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"runner_type": "OnPolicyRunner", "runner": {"stage_index": 0,
                "current_iteration": 4}}, checkpoint)
    return run


def test_fork_archives_input_and_uses_independent_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = source_run(tmp_path)
    old_files = {p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in source.rglob("*") if p.is_file()}
    app = ApplicationEntry("app", STAMP)
    try:
        assert app.save_dir == Path("checkpoints") / f"app_{STAMP}"
        monkeypatch.setattr(app.stage_manager, "train", lambda: None)
        app.train()
        fork = app.save_dir
        assert fork != source
        assert re.fullmatch(r"app_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", fork.name)
        assert app.train_time.strftime("%Y-%m-%d_%H-%M-%S") == fork.name.removeprefix("app_")
        assert (fork / "checkpoints" / "resume.pt").is_file()
        manifest = json.loads((fork / "resume_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "ready"
        assert manifest["mode"] == "fork"
        assert manifest["run_time"] == app.train_time.strftime("%Y-%m-%d_%H-%M-%S")
        assert manifest["current_iteration"] == 4
        assert "source_sha256" not in manifest
        resume = app.stage_manager._prepare_training_stage()
        assert resume is not None
        assert resume.checkpoint_path == fork / "checkpoints" / "resume.pt"
        assert (fork / "configs" / "runners").is_dir()
        assert old_files == {p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source.rglob("*") if p.is_file()}
    finally:
        app.close()
    source.rename(tmp_path / "detached_source")
    with ApplicationEntry("app", fork.name.removeprefix("app_"), resume_mode="inplace") as reopened:
        info = reopened.stage_manager._prepare_training_stage()
        assert info is not None
        assert info.checkpoint_path == (fork / "checkpoints" / "resume.pt").resolve()
        component = reopened.stage_manager._get_current_component()
        assert component.runner is not None and component.runner.config.is_file()


def test_reopened_fork_prefers_own_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_run(tmp_path)
    with ApplicationEntry("app", STAMP) as app:
        app._prepare_fork()
        fork = app.save_dir
    run_name = fork.name.removeprefix("app_")
    with ApplicationEntry("app", run_name, resume_mode="inplace") as reopened:
        resume = reopened.stage_manager._prepare_training_stage()
        assert resume is not None
        assert resume.checkpoint_path == (fork / "checkpoints" / "resume.pt").resolve()
    own = fork / "checkpoints" / "stage_000" / "latest.pt"
    own.parent.mkdir(parents=True)
    torch.save({"runner": {"stage_index": 0, "current_iteration": 6}}, own)
    with ApplicationEntry("app", run_name, resume_mode="inplace") as reopened:
        resume = reopened.stage_manager._prepare_training_stage()
        assert resume is not None
        assert resume.checkpoint_path == own.resolve()


def test_explicit_numbered_checkpoint_ignores_latest_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = source_run(tmp_path)
    numbered = source / "checkpoints" / "stage_000" / "iteration_0002.pt"
    torch.save({"runner": {"stage_index": 0, "current_iteration": 2}}, numbered)
    (numbered.parent / "stage_completed").touch()
    with ApplicationEntry("app", STAMP, resume_mode="inplace", checkpoint="checkpoints/stage_000/iteration_0002.pt") as app:
        info = app.stage_manager._prepare_training_stage()
        assert info is not None
        assert info.checkpoint_path == numbered
        assert info.stage_completed is False


def test_invalid_time_fails_without_creating_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Invalid train_time"):
        ApplicationEntry("app", "bad-time")
    assert not (tmp_path / "checkpoints").exists()


def test_failed_archive_does_not_publish_or_switch_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = source_run(tmp_path)
    with ApplicationEntry("app", STAMP) as app:
        original_context = app.context
        archive = app._archive_configs

        def fail_archive(destination: Path) -> dict:
            archive(destination)
            raise OSError("simulated interrupted copy")

        with monkeypatch.context() as patch:
            patch.setattr(app, "_archive_configs", fail_archive)
            with pytest.raises(OSError, match="interrupted copy"):
                app.train()

        assert app.load_dir.resolve() == source
        assert app.save_dir.resolve() == source
        assert app.context is original_context
        assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir() if not p.name.startswith(".")) == [source.name]
        pending = list((tmp_path / "checkpoints").glob(".*.preparing"))
        assert len(pending) == 1
        manifest = json.loads((pending[0] / "resume_manifest.json").read_text())
        assert manifest["status"] == "preparing"
        with pytest.raises(FileExistsError):
            app._prepare_fork()
        assert app.load_dir.resolve() == source
        assert app.save_dir.resolve() == source
        assert (pending[0] / "checkpoints" / "resume.pt").is_file()


def test_checkpoint_status_overrides_stale_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = source_run(tmp_path)
    latest = source / "checkpoints" / "stage_000" / "latest.pt"
    (latest.parent / "stage_completed").touch()
    torch.save({"runner": {"stage_index": 0, "current_iteration": 3,
                           "checkpoint_version": 2, "stage_completed": False}}, latest)
    with ApplicationEntry("app", STAMP, resume_mode="inplace") as app:
        info = app.stage_manager._prepare_training_stage()
        assert info is not None and info.stage_completed is False


def test_fork_ignores_directory_only_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = source_run(tmp_path)
    latest = source / "checkpoints" / "stage_000" / "latest.pt"
    (latest.parent / "stage_completed").touch()

    with ApplicationEntry("app", STAMP) as app:
        assert app._resume_selection is not None
        assert app._resume_selection.stage_completed is False
        app._prepare_fork()
        assert app.stage_manager.resume_selection is not None
        assert app.stage_manager.resume_selection.stage_completed is False
