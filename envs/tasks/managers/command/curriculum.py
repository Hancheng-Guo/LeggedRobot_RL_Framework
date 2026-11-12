"""Resolve explicit command curriculum references."""
from typing import Any

from .terms import BaseCommandTerm


def curriculum_reference(
    cls: type[BaseCommandTerm],
    params: dict[str, Any],
) -> str | None:

    if not getattr(cls, "uses_curriculum", False):
        return None
    if "group" in params:
        raise ValueError(
            "Curriculum command sampling uses one space per instance; remove 'group'."
        )
    name = params.get("curriculum")
    if not isinstance(name, str) or not name:
        raise ValueError(
            "Curriculum command sampling requires a non-empty 'curriculum' reference."
        )
    return name
