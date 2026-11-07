from typing import TypeVar

from .base import BaseRandomizationTerm
from utils import camel_to_snake


RANDOMIZATION_CLASS_MAP: dict[str, type[BaseRandomizationTerm]] = {}
RandomizationTermType = TypeVar("RandomizationTermType", bound=BaseRandomizationTerm)


def register_randomization(
        cls: type[RandomizationTermType]
    ) -> type[RandomizationTermType]:
    # Keep the existing terrain YAML key while using a descriptive class name.
    name = camel_to_snake(cls.__name__.removesuffix("RandomizationTerm"))
    if name in RANDOMIZATION_CLASS_MAP:
        raise ValueError(f"Randomization term '{name}' is already registered.")
    RANDOMIZATION_CLASS_MAP[name] = cls
    return cls


def get_randomization_class(name: str) -> type[BaseRandomizationTerm]:
    if name not in RANDOMIZATION_CLASS_MAP:
        raise ValueError(f"Unknown randomization term '{name}'.")
    return RANDOMIZATION_CLASS_MAP[name]
