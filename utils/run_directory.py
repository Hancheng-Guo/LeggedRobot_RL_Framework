"""Select a training run directory by path or timestamp."""

import re
from pathlib import Path


RUN_TIMESTAMP = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def select_run_dir(
    checkpoints_dir: Path,
    value: str = ""
) -> Path:
    """Resolve a named run, or the run with the largest timestamp suffix."""
    
    if value.strip():
        requested = Path(value.strip()).expanduser()
        if requested.is_absolute():
            run_dir = requested
        elif requested.parts[0] == checkpoints_dir.name:
            run_dir = checkpoints_dir.parent / requested
        else:
            run_dir = checkpoints_dir / requested
    else:
        candidates = (
            (match.group(1), directory)
            for directory in checkpoints_dir.iterdir()
            if directory.is_dir()
            if (match := RUN_TIMESTAMP.search(directory.name)) is not None
        )
        try:
            _, run_dir = max(candidates)
        except ValueError as exc:
            raise FileNotFoundError(
                f"No dated run directory found in {checkpoints_dir}."
            ) from exc

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir.resolve()
