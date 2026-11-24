from typing import TypeVar

from .base import BaseSimulatorStateTerm
from utils import camel_to_snake


SimulatorStateTermType = TypeVar("SimulatorStateTermType", bound=BaseSimulatorStateTerm)
SIMULATOR_STATE_TERM_CLASSES: dict[str, type[BaseSimulatorStateTerm]] = {}


def register_simulator_state_term(cls: type[SimulatorStateTermType]) -> type[SimulatorStateTermType]:
    name = camel_to_snake(cls.__name__)
    if name in SIMULATOR_STATE_TERM_CLASSES:
        raise ValueError(f"Simulator state term '{name}' is already registered.")
    SIMULATOR_STATE_TERM_CLASSES[name] = cls
    return cls


def get_simulator_state_term_class(name: str) -> type[BaseSimulatorStateTerm]:
    if name not in SIMULATOR_STATE_TERM_CLASSES:
        raise ValueError(f"Unknown simulator state term '{name}'.")
    return SIMULATOR_STATE_TERM_CLASSES[name]
