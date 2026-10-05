import sys
import pytest
from pathlib import Path
from types import ModuleType

from tools import play_checkpoint


def test_run_identity_parses_app_name_and_time() -> None:
    run_dir = Path("checkpoints/unitree_go1_mujoco_cpu_velocity_2026-10-01_12-30-05")
    assert play_checkpoint.run_identity(run_dir) == (
        "unitree_go1_mujoco_cpu_velocity",
        "2026-10-01_12-30-05",
    )


@pytest.mark.parametrize("selected", ["", "robot_2026-10-01_12-30-05"])
def test_main_plays_selected_run_and_closes_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    run_dir = tmp_path / "checkpoints" / "robot_2026-10-01_12-30-05"
    run_dir.mkdir(parents=True)
    calls: list[object] = []

    class FakeApplication:
        def __init__(self, app_name: str, train_time: str) -> None:
            calls.append((app_name, train_time))

        def play(self, *, num_plays: int, formats: str) -> None:
            assert (num_plays, formats) == (3, "gif")
            calls.append("play")
            raise RuntimeError("play failed")

        def close(self) -> None:
            calls.append("close")

    fake_app = ModuleType("app")
    fake_app.ApplicationEntry = FakeApplication  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app", fake_app)
    monkeypatch.setattr(play_checkpoint, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="play failed"):
        play_checkpoint.main([selected])

    assert calls == [
        ("robot", "2026-10-01_12-30-05"),
        "play",
        "close",
    ]


def test_main_prints_each_new_video_without_reporting_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "checkpoints" / "robot_2026-10-01_12-30-05"
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "existing.gif").write_bytes(b"old")

    class FakeApplication:
        def __init__(self, app_name: str, train_time: str) -> None:
            assert (app_name, train_time) == ("robot", "2026-10-01_12-30-05")

        def play(self, *, num_plays: int, formats: str) -> None:
            assert (num_plays, formats) == (3, "gif")
            (video_dir / "first.gif").write_bytes(b"gif")
            (video_dir / "second.mp4").write_bytes(b"mp4")

        def close(self) -> None:
            pass

    fake_app = ModuleType("app")
    fake_app.ApplicationEntry = FakeApplication  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app", fake_app)
    monkeypatch.setattr(play_checkpoint, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)

    play_checkpoint.main([run_dir.name])

    output = capsys.readouterr().out
    assert f"Playing checkpoint: {run_dir}" in output
    assert f"Saved playback: {video_dir / 'first.gif'}" in output
    assert f"Saved playback: {video_dir / 'second.mp4'}" in output
    assert "existing.gif" not in output
