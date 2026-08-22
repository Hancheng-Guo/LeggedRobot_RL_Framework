from typing import TypeVar

from envs.tasks.managers.termination.terms.base import BaseTerminationTerm
from utils.string import camel_to_snake


TERMINATION_CLASS_MAP: dict[str, type[BaseTerminationTerm]] = {}


TerminationTermType = TypeVar(
    "TerminationTermType",
    bound=BaseTerminationTerm,
)


def register_termination(
    cls: type[TerminationTermType],
) -> type[TerminationTermType]:
    
    name = camel_to_snake(cls.__name__)

    if name in TERMINATION_CLASS_MAP:
        raise ValueError(f"Termination term '{name}' is already registered.")
    
    TERMINATION_CLASS_MAP[name] = cls

    return cls


def get_termination_class(name: str) -> type[BaseTerminationTerm]:

    if name not in TERMINATION_CLASS_MAP:
        raise ValueError(f"Unknown termination term '{name}'.")
    
    return TERMINATION_CLASS_MAP[name]
