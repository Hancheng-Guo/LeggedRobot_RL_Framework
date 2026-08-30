from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any


def resolve_metric_name(
    info: Mapping[str, Any],
    pattern: str,
    owner: str,
) -> str:
    matches = [
        name
        for name in info
        if fnmatchcase(name, pattern)
    ]
    if not matches:
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

