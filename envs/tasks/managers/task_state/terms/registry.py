from typing import TypeVar

from .base import BaseTaskStateTerm
from utils import camel_to_snake


TASK_STATE_CLASS_MAP: dict[str, type[BaseTaskStateTerm]] = {}
TaskStateTermType = TypeVar("TaskStateTermType", bound=BaseTaskStateTerm)


def register_task_state(cls: type[TaskStateTermType]) -> type[TaskStateTermType]:
    name = camel_to_snake(cls.__name__)
    if name in TASK_STATE_CLASS_MAP:
        raise ValueError(f"Task state term '{name}' is already registered.")
    TASK_STATE_CLASS_MAP[name] = cls
    return cls


def get_task_state_class(name: str) -> type[BaseTaskStateTerm]:
    if name not in TASK_STATE_CLASS_MAP:
        raise ValueError(f"Unknown task state term '{name}'.")
    return TASK_STATE_CLASS_MAP[name]
