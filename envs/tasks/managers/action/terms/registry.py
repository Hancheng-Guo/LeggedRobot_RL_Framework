from envs.tasks.managers.action.terms.base import BaseActionTerm
from utils.string import camel_to_snake


ACTION_CLASS_MAP: dict[str, type[BaseActionTerm]] = {}


def register_action(
    cls: type[BaseActionTerm],
) -> type[BaseActionTerm]:

    name = camel_to_snake(cls.__name__)

    if name in ACTION_CLASS_MAP:
        raise ValueError(
            f"Action class '{name}' already registered."
        )

    ACTION_CLASS_MAP[name] = cls

    return cls


def get_action_class(action_name: str):

    cls = ACTION_CLASS_MAP.get(action_name, None)

    if cls is None:
        raise ValueError(
            f"Unknown action class: '{action_name}'."
        )

    return cls

