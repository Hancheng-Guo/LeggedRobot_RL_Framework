from envs.tasks.managers.command.terms.base import BaseCommandTerm


COMMAND_CLASS_MAP: dict[str, type[BaseCommandTerm]] = {}


def register_command(
    cls: type[BaseCommandTerm],
) -> type[BaseCommandTerm]:

    name = cls.__name__

    if name in COMMAND_CLASS_MAP:
        raise ValueError(
            f"Command type '{name}' is already registered."
        )

    COMMAND_CLASS_MAP[name] = cls

    return cls


def get_command_class(
    name: str,
) -> type[BaseCommandTerm]:

    return COMMAND_CLASS_MAP[name]