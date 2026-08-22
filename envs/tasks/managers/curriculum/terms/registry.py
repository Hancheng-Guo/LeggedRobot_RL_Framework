from typing import TypeVar

from envs.tasks.managers.curriculum.terms.base import BaseCurriculumTerm
from utils.string import camel_to_snake


CURRICULUM_CLASS_MAP: dict[str, type[BaseCurriculumTerm]] = {}


CurriculumTermType = TypeVar(
    "CurriculumTermType",
    bound=BaseCurriculumTerm,
)


def register_curriculum(
    cls: type[CurriculumTermType],
) -> type[CurriculumTermType]:
    
    name = camel_to_snake(cls.__name__)

    if name in CURRICULUM_CLASS_MAP:
        raise ValueError(f"Curriculum term '{name}' is already registered.")
    
    CURRICULUM_CLASS_MAP[name] = cls

    return cls


def get_curriculum_class(name: str) -> type[BaseCurriculumTerm]:

    if name not in CURRICULUM_CLASS_MAP:
        raise ValueError(f"Unknown curriculum term '{name}'.")
    
    return CURRICULUM_CLASS_MAP[name]
