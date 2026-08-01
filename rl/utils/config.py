from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """laod YAML config file."""

    config_path = Path(path)

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