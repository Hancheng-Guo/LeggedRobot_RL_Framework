"""Load the independent command profile for ROS 2 keyboard playback."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from utils import load_yaml


TWIST_FIELDS = {
    "linear.x", "linear.y", "linear.z",
    "angular.x", "angular.y", "angular.z",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RosCommandMapping:
    field: str
    positive_key: str
    negative_key: str
    speed: float


@dataclass(frozen=True)
class HumanRos2Config:
    path: Path
    topic: str
    ros_commands: Mapping[str, RosCommandMapping]
    fixed_commands: Mapping[str, tuple[float, ...]]


def load_human_ros2_config(config_path: str | Path) -> HumanRos2Config:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    raw = load_yaml(path)
    if set(raw) != {"topic", "ros_commands", "fixed_commands"}:
        raise ValueError(f"{path}: expected topic, ros_commands, fixed_commands.")

    topic = raw["topic"]
    if not isinstance(topic, str) or not topic.startswith("/"):
        raise ValueError(f"{path}: topic must be an absolute ROS topic.")
    raw_ros = raw["ros_commands"]
    raw_fixed = raw["fixed_commands"]
    if not isinstance(raw_ros, dict) or not raw_ros:
        raise ValueError(f"{path}: ros_commands must be a non-empty mapping.")
    if not isinstance(raw_fixed, dict):
        raise ValueError(f"{path}: fixed_commands must be a mapping.")

    ros_commands: dict[str, RosCommandMapping] = {}
    used_fields: set[str] = set()
    used_keys: set[str] = set()
    for name, entry in raw_ros.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError(f"{path}: each ROS command needs a name and mapping.")
        if set(entry) != {"field", "positive_key", "negative_key", "speed"}:
            raise ValueError(f"{path}: ROS command {name!r} needs field, keys, speed.")
        field = entry["field"]
        if not isinstance(field, str) or field not in TWIST_FIELDS or field in used_fields:
            raise ValueError(f"{path}: invalid or duplicate Twist field {field!r}.")
        used_fields.add(field)
        keys = []
        for key_name in ("positive_key", "negative_key"):
            key = entry[key_name]
            if not isinstance(key, str) or not key or key != key.lower() or key == "space" or key in used_keys:
                raise ValueError(f"{path}: invalid or duplicate {key_name} for {name!r}.")
            used_keys.add(key)
            keys.append(key)
        speed = entry["speed"]
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError(f"{path}: speed for {name!r} must be positive and finite.")
        ros_commands[name] = RosCommandMapping(field, keys[0], keys[1], float(speed))

    fixed_commands: dict[str, tuple[float, ...]] = {}
    for name, value in raw_fixed.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: fixed command names must be non-empty strings.")
        values = value if isinstance(value, list) else [value]
        if not values or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in values
        ):
            raise ValueError(f"{path}: fixed command {name!r} must contain finite numbers.")
        fixed_commands[name] = tuple(float(item) for item in values)

    return HumanRos2Config(path, topic, ros_commands, fixed_commands)
