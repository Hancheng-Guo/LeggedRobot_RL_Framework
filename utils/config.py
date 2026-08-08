from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """load YAML config file."""

    config_path = Path(path)

    if config_path.suffix == '':
        config_path = config_path.with_suffix(".yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"File {config_path.resolve()} cannot be found."
        )

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if config is None:
        raise ValueError(f"File {config_path.resolve()} is empty.")

    if not isinstance(config, dict):
        raise TypeError("Root node must be a Dict.")

    return config


def get_yaml_value(
    yaml_path: Path | str,
    key_path: str,
    default: Any = None,
    required: bool = False,
) -> Any:

    config = load_yaml(Path(yaml_path))

    current = config

    for key in key_path.split("."):

        if isinstance(current, dict):
            if key not in current:
                if required:
                    raise KeyError(f"Missing yaml field: {key_path}")
                return default

            current = current[key]

        elif isinstance(current, list):
            if not key.isdigit():
                raise TypeError(
                    f"Expected list index, got '{key}' "
                    f"in path '{key_path}'"
                )

            index = int(key)

            if index >= len(current) or index < 0:
                if required:
                    raise IndexError(
                        f"List index out of range: {key_path}"
                    )
                return default

            current = current[index]

        else:
            if required:
                raise TypeError(
                    f"Cannot access '{key}' from "
                    f"{type(current).__name__}"
                )
            return default

    return current