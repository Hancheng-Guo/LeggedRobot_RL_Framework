"""The playback profile defines both ROS input and fixed command values."""

from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

from runners.utils.human_ros2_config import PROJECT_ROOT, RosCommandMapping, load_human_ros2_config
from runners.utils.ros2_keyboard import Ros2KeyboardSession
from utils import load_yaml


def test_phase_profile_maps_ros_and_fixed_commands() -> None:
    config = load_human_ros2_config("configs/playback/human_ros2/locomotion_unitree_go1_phase.yaml")
    assert config.topic == "/cmd_vel"
    assert config.ros_commands["lin_vel_x"].field == "linear.x"
    assert config.ros_commands["ang_vel_z"].positive_key == "q"
    assert config.fixed_commands == {
        "foot_phase_real": (1.0, 1.0, 1.0, 1.0),
        "foot_phase_imag": (0.0, 0.0, 0.0, 0.0),
    }

    session = object.__new__(Ros2KeyboardSession)
    session._config = config
    session._values = {name: 0.0 for name in config.ros_commands}
    session._last_message_at = None
    message = SimpleNamespace(
        linear=SimpleNamespace(x=1.2, y=-0.2, z=0.0),
        angular=SimpleNamespace(x=0.0, y=0.0, z=0.5),
    )
    session._on_message(message)
    assert session.values() == {
        "lin_vel_x": 0.8,
        "lin_vel_y": -0.2,
        "ang_vel_z": 0.5,
    }


def test_all_task_profiles_load() -> None:
    for name in (
        "locomotion_anymal_c_phase",
        "locomotion_unitree_go1_velocity",
        "locomotion_unitree_go1_standing",
    ):
        config = load_human_ros2_config(f"configs/playback/human_ros2/{name}.yaml")
        assert set(config.ros_commands) == {"lin_vel_x", "lin_vel_y", "ang_vel_z"}


def test_profiles_cover_their_task_commands() -> None:
    task_dir = PROJECT_ROOT / "configs" / "tasks"
    profile_dir = PROJECT_ROOT / "configs" / "playback" / "human_ros2"
    for profile_path in profile_dir.glob("*.yaml"):
        task_path = task_dir / profile_path.name
        config = load_human_ros2_config(profile_path)
        task = load_yaml(task_path)
        terms = task["command_manager_config"]["terms"]
        assert set(terms) == set(config.ros_commands) | set(config.fixed_commands)
        for name, value in config.fixed_commands.items():
            assert len(value) == terms[name]["params"].get("dim", 1)


def test_ros_message_mapping_comes_from_profile() -> None:
    config = load_human_ros2_config("configs/playback/human_ros2/locomotion_unitree_go1_phase.yaml")
    config = replace(
        config,
        ros_commands={"body_height": RosCommandMapping("linear.z", "r", "f", 0.2)},
    )
    session = object.__new__(Ros2KeyboardSession)
    session._config = config
    session._values = {"body_height": 0.0}
    session._last_message_at = None
    message = SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0, z=0.3))
    session._on_message(message)
    assert session.values() == {"body_height": 0.2}


def test_poll_uses_session_executor() -> None:
    config = load_human_ros2_config("configs/playback/human_ros2/locomotion_unitree_go1_phase.yaml")
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
    original = PROJECT_ROOT / "configs/playback/human_ros2/locomotion_unitree_go1_phase.yaml"
    custom = tmp_path / "my_keyboard_profile.yaml"
    custom.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    config = load_human_ros2_config(custom)
    assert config.path == custom.resolve()
