from dataclasses import dataclass
from typing import Any
from pathlib import Path

from utils.config import load_yaml
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
    ) -> ComponentInfo:

    component = component_dict.get(component_name, None)

    if component is None:
        return None

    config_name = component.get("config", None)
    config_dir = component.get(
        "config_dir",
        load_yaml("./configs/base.yaml").get(
            f"{component_name}_config_dir"
        )
    )
    config_path = component.get("config_path", None)
    
    config_path = fill_path(
            file_name=config_name,
            file_dir=config_dir,
            file_path=config_path,
        )

    if config_path.suffix == '':
        config_path = config_path.with_suffix(".yaml")

    return ComponentInfo(
        type=component.get("type", None),
        config=config_path,
    )


def create_component(component_dict: dict[str, Any]) -> Component:

    return Component(
        runner=_create_component_info(component_dict, "runner"),
        algorithm=_create_component_info(component_dict, "algorithm"),
        model=_create_component_info(component_dict, "model"),
        environment=_create_component_info(component_dict, "environment"),
        simulator=_create_component_info(component_dict, "simulator"),
        task=_create_component_info(component_dict, "task"),
    )
    