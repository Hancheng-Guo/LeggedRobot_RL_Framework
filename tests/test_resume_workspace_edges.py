"""Regression checks for isolated historical training workspaces."""

import json
from datetime import datetime, tzinfo
from pathlib import Path

import pytest
import torch
import yaml

from app.application_entry import ApplicationEntry
from app.utils.run_workspace import select_checkpoint


STAMP = "2026-01-02_03-04-05"


def make_source(tmp_path: Path, *, components: dict | None = None) -> Path:
    run = tmp_path / "checkpoints" / f"app_{STAMP}"
    (run / "configs").mkdir(parents=True)
    (run / "configs" / "runners").mkdir()
    (run / "configs" / "runners" / "runner.yaml").write_text("rollout_length: 8\n", encoding="utf-8")
    config = {
        "runtime": {"device": "cpu", "dtype": "float32", "num_threads": 1,
                    "seed": 5, "deterministic_ops": False},
        "logging": {"console": False},
        "stage": [{"one": {"max_iterations": 10, "transition": [True]}}],
        "component": components if components is not None else {"runner": [{"type": "on_policy", "config": "runner"}]},
    }
    (run / "configs" / "app.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    latest = run / "checkpoints" / "stage_000" / "latest.pt"
    latest.parent.mkdir(parents=True)
    torch.save({"runner": {"stage_index": 0, "current_iteration": 4}}, latest)
    return run


@pytest.mark.parametrize("state", ["missing", "preparing"])
def test_archived_checkpoint_does_not_depend_on_manifest(tmp_path: Path, state: str) -> None:
    run = tmp_path / "fork"
    archived = run / "checkpoints" / "resume.pt"
    archived.parent.mkdir(parents=True)
    torch.save({"runner": {"stage_index": 0, "current_iteration": 4}}, archived)
    if state == "preparing":
        (run / "resume_manifest.json").write_text(
            json.dumps({"status": "preparing", "stage_index": 0}), encoding="utf-8"
        )
    selected = select_checkpoint(run, 1)
    assert selected.checkpoint == archived.resolve()
    assert selected.stage_index == 0


def test_checkpoint_stage_must_match_stage_directory(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "stage_001" / "latest.pt"
    path.parent.mkdir(parents=True)
    torch.save({"runner": {"stage_index": 0, "current_iteration": 4}}, path)

    with pytest.raises(ValueError, match="does not match directory stage"):
        select_checkpoint(
            tmp_path,
            2,
        )


def test_checkpoint_without_stage_index_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "resume.pt"
    path.parent.mkdir(parents=True)
    torch.save({"runner": {"current_iteration": 4}}, path)

    with pytest.raises(ValueError, match="Invalid checkpoint stage_index None"):
        select_checkpoint(
            tmp_path,
            1,
        )


@pytest.mark.parametrize("directory", ["stage_1", "stage_0001", "stage_abc"])
def test_explicit_checkpoint_rejects_nonstandard_stage_directory(
    tmp_path: Path, directory: str,
) -> None:
    path = tmp_path / "checkpoints" / directory / "latest.pt"
    path.parent.mkdir(parents=True)
    torch.save({"runner": {"stage_index": 1, "current_iteration": 4}}, path)

    with pytest.raises(ValueError, match="Invalid checkpoint stage directory"):
        select_checkpoint(tmp_path, 2, selected=path)


def test_automatic_selection_rejects_nonstandard_stage_directory(tmp_path: Path) -> None:
    invalid = tmp_path / "checkpoints" / "stage_1" / "latest.pt"
    invalid.parent.mkdir(parents=True)
    torch.save({"runner": {"stage_index": 1, "current_iteration": 4}}, invalid)
    resume = tmp_path / "checkpoints" / "resume.pt"
    torch.save({"runner": {"stage_index": 0, "current_iteration": 2}}, resume)

    with pytest.raises(ValueError, match="Invalid checkpoint stage directory"):
        select_checkpoint(tmp_path, 2)


def test_external_same_name_configs_survive_source_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    external_a = tmp_path / "external_a" / "runner.yaml"
    external_b = tmp_path / "external_b" / "runner.yaml"
    external_a.parent.mkdir()
    external_b.parent.mkdir()
    external_a.write_text("rollout_length: 8\n", encoding="utf-8")
    external_b.write_text("rollout_length: 16\n", encoding="utf-8")
    source = make_source(tmp_path, components={"runner": [
        {"type": "on_policy", "config_path": str(external_a)},
        {"type": "on_policy", "config_path": str(external_b)},
    ]})
    config_path = source / "configs" / "app.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["stage"].append({"two": {"max_iterations": 10, "transition": [True]}})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with ApplicationEntry("app", STAMP) as app:
        app._prepare_fork()
        fork = app.save_dir
    source.rename(tmp_path / "moved_source")
    external_a.rename(external_a.with_name("moved_runner_a.yaml"))
    external_b.rename(external_b.with_name("moved_runner_b.yaml"))

    with ApplicationEntry("app", fork.name.removeprefix("app_"), resume_mode="inplace") as reopened:
        entries = reopened.config["component"]["runner"]
        assert entries[0]["config"] != entries[1]["config"]
        for entry, expected in zip(entries, (8, 16)):
            path = reopened.load_dir / "configs" / "runners" / entry["config"]
            assert yaml.safe_load(path.read_text(encoding="utf-8"))["rollout_length"] == expected


def test_evaluation_does_not_create_fork(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = make_source(tmp_path)
    with ApplicationEntry("app", STAMP) as app:
        monkeypatch.setattr(app.stage_manager, "test", lambda **kwargs: None)
        monkeypatch.setattr(app.stage_manager, "play", lambda **kwargs: None)
        app.test(num_episodes=1)
        app.play(num_steps=1)
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == [source.name]
    assert not (source / "logs").exists()


def test_same_second_fork_name_collision_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    make_source(tmp_path)
    fixed = datetime(2026, 2, 3, 23, 59, 58)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return fixed

    monkeypatch.setattr("app.application_entry.datetime", FixedDatetime)
    with ApplicationEntry("app", STAMP) as app:
        app._prepare_fork()
        fork = app.save_dir
        assert fork.name == "app_2026-02-03_23-59-58"
        manifest = json.loads((fork / "resume_manifest.json").read_text(encoding="utf-8"))
        assert manifest["run_time"] == "2026-02-03_23-59-58"
    with ApplicationEntry("app", STAMP) as app:
        with pytest.raises(FileExistsError, match="already exists"):
            app._prepare_fork()
    with ApplicationEntry("app", fork.name.removeprefix("app_"), resume_mode="inplace") as reopened:
        assert reopened.stage_manager._prepare_training_stage() is not None


def test_existing_fork_staging_directory_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    make_source(tmp_path)
    preparing_dir = tmp_path / "checkpoints" / ".app_2026-02-03_23-59-58.preparing"
    preparing_dir.mkdir()

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return datetime(2026, 2, 3, 23, 59, 58)

    monkeypatch.setattr("app.application_entry.datetime", FixedDatetime)
    with ApplicationEntry("app", STAMP) as app:
        with pytest.raises(FileExistsError):
            app._prepare_fork()
    assert preparing_dir.is_dir()


def test_fork_suffix_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="expected YYYY-MM-DD_HH-MM-SS"):
        ApplicationEntry("app", "2026-01-02_03-04-05_fork_20260102_030405_000")
    assert not (tmp_path / "checkpoints").exists()
