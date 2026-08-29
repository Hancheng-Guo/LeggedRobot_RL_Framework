import torch
from collections.abc import Sequence

from envs.simulators.utils.context import ModelContext
from envs.tasks.utils.context import TaskContext


def geom_ids_from_names(
    model_context: ModelContext,
    names: Sequence[str],
    fallback_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    
    if not names:
        if fallback_ids is None:
            raise ValueError("Geom names must be provided.")
        if fallback_ids.numel() == 0:
            raise ValueError("No fallback geom IDs are available.")
        return fallback_ids

    name_to_id = {
        name: geom_id
        for geom_id, name in enumerate(model_context.geom_names)
        if name is not None
    }
    unknown_names = set(names) - name_to_id.keys()
    if unknown_names:
        raise ValueError(f"Unknown geom name(s): {sorted(unknown_names)}.")

    return torch.tensor(
        [name_to_id[name] for name in names],
        dtype=torch.long,
        device=(
            fallback_ids.device
            if fallback_ids is not None
            else model_context.geom_body_ids.device
        ),
    )


def command_vector(
    task_context: TaskContext,
    names: Sequence[str] | None,
) -> torch.Tensor:
    
    if names is None:
        names = tuple(task_context.command)
    else:
        missing_names = tuple(
            name
            for name in names
            if name not in task_context.command
        )
        if missing_names:
            raise ValueError(f"Unknown command name(s): {missing_names}.")

    values = [task_context.command[name] for name in names]
    if values:
        return torch.cat(values, dim=-1)
    
    raise RuntimeError("Empty command vector.")
