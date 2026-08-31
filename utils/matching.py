from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any, Literal, overload


@overload
def resolve_metric_name(
    info: Mapping[str, Any],
    pattern: str,
    owner: str,
    require_match: Literal[True] = True,
) -> str: ...


@overload
def resolve_metric_name(
    info: Mapping[str, Any],
    pattern: str,
    owner: str,
    require_match: Literal[False],
) -> str | None: ...


def resolve_metric_name(
    info: Mapping[str, Any],
    pattern: str,
    owner: str,
    require_match: bool = True,
) -> str | None:
    matches = [
        name
        for name in info
        if fnmatchcase(name, pattern)
    ]
    if not matches:
        if not require_match:
            return None
        raise KeyError(
            f"Training info has no metric matching {owner} "
            f"pattern {pattern!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{owner} pattern {pattern!r} is ambiguous; "
            f"matched: {', '.join(matches)}."
        )
    return matches[0]
