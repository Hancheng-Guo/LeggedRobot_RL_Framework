"""The playback profile defines both ROS input and fixed command values."""

from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from runners.utils.human_ros2_config import PROJECT_ROOT, RosCommandMapping, load_human_ros2_config
from runners.utils.ros2_keyboard import Ros2KeyboardSession, _KeyboardGears
from utils import load_yaml


def test_phase_profile_maps_ros_and_fixed_commands() -> None:
    config = load_human_ros2_config("configs/playback/locomotion_phase.yaml")
    assert config.topic == "/cmd_vel"
    assert config.ros_commands["lin_vel_x"].field == "linear.x"
    assert config.ros_commands["lin_vel_x"].levels[0] == -2.5
    assert config.ros_commands["lin_vel_x"].levels[-1] == 2.5
    assert config.fixed_commands == {
        "foot_phase_real": (1.0, 1.0, 1.0, 1.0),
        "foot_phase_imag": (0.0, 0.0, 0.0, 0.0),
    }

    session = object.__new__(Ros2KeyboardSession)
    session._config = config
    session._values = {name: 0.0 for name in config.ros_commands}
    session._last_message_at = None
    message = SimpleNamespace(
        linear=SimpleNamespace(x=3.0, y=-0.2, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=0.5),
    )
    session._on_message(message)
    assert session.values() == {
        "lin_vel_x": 2.5,
        "lin_vel_y": -0.2,
        "ang_vel_z": 0.5,
    }


def test_shared_playback_profiles_load() -> None:
    lateral_levels = {
        "locomotion_phase": (-0.4, -0.2, 0.0, 0.2, 0.4),
        "locomotion_velocity": (-0.5, -0.25, 0.0, 0.25, 0.5),
    }
    for name in ("locomotion_phase", "locomotion_velocity"):
        config = load_human_ros2_config(f"configs/playback/{name}.yaml")
        assert set(config.ros_commands) == {"lin_vel_x", "lin_vel_y", "ang_vel_z"}
        assert config.ros_commands["lin_vel_x"].levels == (
            -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
            0.5, 1.0, 1.5, 2.0, 2.5,
        )
        assert config.ros_commands["lin_vel_y"].levels == lateral_levels[name]
        assert config.ros_commands["ang_vel_z"].levels == (
            -1.5, -0.75, 0.0, 0.75, 1.5,
        )
        assert config.ros_commands["ang_vel_z"].positive_key == "q"
        assert config.ros_commands["ang_vel_z"].negative_key == "e"


def test_profiles_cover_their_task_commands() -> None:
    task_dir = PROJECT_ROOT / "configs" / "tasks"
    profile_dir = PROJECT_ROOT / "configs" / "playback"
    profile_tasks = {
        "locomotion_phase": (
            "locomotion_anymal_c_phase", "locomotion_unitree_go1_phase",
        ),
        "locomotion_velocity": ("locomotion_unitree_go1_velocity",),
    }
    assert {path.stem for path in profile_dir.glob("*.yaml")} == set(profile_tasks)
    for profile_name, task_names in profile_tasks.items():
        profile_path = profile_dir / f"{profile_name}.yaml"
        config = load_human_ros2_config(profile_path)
        for task_name in task_names:
            task = load_yaml(task_dir / f"{task_name}.yaml")
            terms = task["command_manager_config"]["terms"]
            assert set(terms) == set(config.ros_commands) | set(config.fixed_commands)
            for name, mapping in config.ros_commands.items():
                term = terms[name]
                params = term["params"]
                maximum = params["max_value"]
                minimum = -maximum if term["type"].startswith("Symmetric") else params["min_value"]
                assert mapping.levels[0] >= minimum
                assert mapping.levels[-1] <= maximum
            for name, value in config.fixed_commands.items():
                assert len(value) == terms[name]["params"].get("dim", 1)


def test_ros_message_mapping_comes_from_profile() -> None:
    config = load_human_ros2_config("configs/playback/locomotion_phase.yaml")
    config = replace(
        config,
        ros_commands={"body_height": RosCommandMapping("linear.z", "r", "f", (-0.2, 0.0, 0.2))},
    )
    session = object.__new__(Ros2KeyboardSession)
    session._config = config
    session._values = {"body_height": 0.0}
    session._last_message_at = None
    message = SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0, z=0.3))
    session._on_message(message)
    assert session.values() == {"body_height": 0.2}


def test_keyboard_gears_step_once_per_press_and_hold_on_release() -> None:
    config = load_human_ros2_config("configs/playback/locomotion_phase.yaml")
    gears = _KeyboardGears(config.ros_commands)

    assert gears.value("lin_vel_x") == 0.0
    assert gears.press("w")
    assert gears.value("lin_vel_x") == 0.5
    assert not gears.press("w")
    assert gears.value("lin_vel_x") == 0.5
    gears.release("w")
    assert gears.value("lin_vel_x") == 0.5
    gears.press("w")
    gears.release("w")
    gears.press("w")
    assert gears.value("lin_vel_x") == 1.5
    gears.press("a")
    assert gears.value("lin_vel_y") == 0.2
    gears.press("space")
    assert gears.value("lin_vel_x") == 0.0
    assert gears.value("lin_vel_y") == 0.0
    gears.reset()
    gears.press("s")
    assert gears.value("lin_vel_x") == -0.5


@pytest.mark.parametrize("levels", [
    [-0.8, 0.4, 0.8],
    [-0.8, 0.0, 0.0, 0.8],
    [0.0, 0.4, 0.8],
    [-0.8, 0.0, float("nan")],
    [-0.8, False, 0.8],
])
def test_invalid_gear_levels_are_rejected(tmp_path: Path, levels: list[float]) -> None:
    original = PROJECT_ROOT / "configs/playback/locomotion_phase.yaml"
    raw = load_yaml(original)
    raw["ros_commands"]["lin_vel_x"]["levels"] = levels
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="levels"):
        load_human_ros2_config(path)


def test_old_speed_mapping_is_rejected(tmp_path: Path) -> None:
    original = PROJECT_ROOT / "configs/playback/locomotion_phase.yaml"
    raw = load_yaml(original)
    entry = raw["ros_commands"]["lin_vel_x"]
    del entry["levels"]
    entry["speed"] = 0.8
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="levels"):
        load_human_ros2_config(path)


def test_poll_uses_session_executor() -> None:
    config = load_human_ros2_config("configs/playback/locomotion_phase.yaml")
    calls: list[float] = []
    session = object.__new__(Ros2KeyboardSession)
    session._config = config
    setattr(session, "_node", object())
    setattr(session, "_executor", SimpleNamespace(spin_once=lambda timeout_sec: calls.append(timeout_sec)))
    session._values = {name: 1.0 for name in config.ros_commands}
    session._last_message_at = None

    session.poll()

    assert calls == [0.0]
    assert session.values() == {name: 0.0 for name in config.ros_commands}


def test_custom_named_profile_path_is_loaded(tmp_path: Path) -> None:
    original = PROJECT_ROOT / "configs/playback/locomotion_phase.yaml"
    custom = tmp_path / "my_keyboard_profile.yaml"
    custom.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    config = load_human_ros2_config(custom)
    assert config.path == custom.resolve()
