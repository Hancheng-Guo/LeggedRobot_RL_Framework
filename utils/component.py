from dataclasses import dataclass
from typing import Any
from pathlib import Path

from utils.path import fill_path


@dataclass(frozen=True)
class ComponentInfo:
    type: str
    config: Path


@dataclass(frozen=True)
class Component:
    runner: ComponentInfo | None
    algorithm: ComponentInfo | None
    model: ComponentInfo | None
    environment: ComponentInfo | None
    simulator: ComponentInfo | None
    task: ComponentInfo | None


def _create_component_info(
        component_dict: dict[str, Any],
        component_name: str,
        load_dir: Path,
    ) -> ComponentInfo | None:

    component = component_dict.get(component_name, None)

    if component is None:
        return None

    config_path = component.get("config_path", None)

    if config_path is None:
        config_name = component.get("config", None)
        config_dir = component.get(
            "config_dir",
            load_dir / "configs" / f"{component_name}s"
        )
        config_path = fill_path(
            file_name=config_name,
            file_dir=config_dir,
        )
    else:
        config_path = fill_path(
            file_path=config_path,
        )

    if config_path.suffix == '':
        config_path = config_path.with_suffix(".yaml")

    return ComponentInfo(
        type=component.get("type", None),
        config=config_path,
    )


def create_component(
    component_dict: dict[str, Any],
    load_dir: Path,
) -> Component:

    return Component(
        runner=_create_component_info(component_dict, "runner", load_dir), 
        algorithm=_create_component_info(component_dict, "algorithm", load_dir), 
        model=_create_component_info(component_dict, "model", load_dir), 
        environment=_create_component_info(component_dict, "environment", load_dir), 
        simulator=_create_component_info(component_dict, "simulator", load_dir), 
        task=_create_component_info(component_dict, "task", load_dir), 
    )
    