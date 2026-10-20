"""The human ROS 2 entry forwards the explicitly selected playback file."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools import play_human_ros2


@pytest.mark.parametrize("config_arg", [
    "custom", "custom.yaml", "configs/playback/human_ros2/custom.yaml",
    "absolute",
])
@pytest.mark.parametrize("run_arg", ["named", "latest"])
def test_main_passes_explicit_playback_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    config_arg: str, run_arg: str,
) -> None:
    run_dir = tmp_path / "checkpoints" / "robot_2026-10-01_12-30-05"
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "configs" / "playback" / "human_ros2" / "custom.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("topic: /cmd_vel\n", encoding="utf-8")
    calls: list[object] = []

    class FakeApplication:
        def __init__(self, app_name: str, train_time: str) -> None:
            calls.append((app_name, train_time))

        def play(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def close(self) -> None:
            calls.append("close")

    fake_app = ModuleType("app")
    fake_app.ApplicationEntry = FakeApplication  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app", fake_app)
    monkeypatch.setattr(play_human_ros2, "PROJECT_ROOT", tmp_path)

    play_human_ros2.main([
        run_dir.name if run_arg == "named" else "",
        "--playback-config", str(config_path) if config_arg == "absolute" else config_arg,
    ])

    assert calls == [
        ("robot", "2026-10-01_12-30-05"),
        {"mode": "human_ros2", "num_steps": None, "num_plays": 1,
         "playback_config": config_path},
        "close",
    ]


def test_main_requires_playback_config() -> None:
    with pytest.raises(SystemExit, match="2"):
        play_human_ros2.main([])
