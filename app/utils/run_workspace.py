"""Checkpoint selection for training, evaluation, and fork preparation."""

from __future__ import annotations

import torch
import re
from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class CheckpointInfo:
    stage_index: int
    current_iteration: int
    stage_completed: bool


@dataclass(frozen=True)
class CheckpointSelection:
    checkpoint: Path
    stage_index: int
    stage_completed: bool


def inspect_checkpoint_metadata(path: Path) -> CheckpointInfo:
    """Read the framework checkpoint metadata without building a runner."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["runner"]

    stage = state.get("stage_index")
    if not isinstance(stage, int) or isinstance(stage, bool) or stage < 0:
        raise ValueError(f"Invalid checkpoint stage_index {stage!r} in {path}.")

    iteration = state.get("current_iteration", -1)
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < -1:
        raise ValueError(f"Invalid checkpoint current_iteration {iteration!r} in {path}.")

    if "stage_completed" in state:
        completed = bool(state["stage_completed"])
    else:
        callbacks = state.get("callbacks", ())
        completed = any(
            entry.get("state", {}).get("stop_training") is True
            for entry in callbacks
            if entry.get("type") == "StageCallback"
        )

    return CheckpointInfo(stage, iteration, completed)


def select_checkpoint(
    run_dir: Path,
    stage_count: int,
    selected: str | Path | None = None,
) -> CheckpointSelection:

    run_dir = run_dir.resolve()
    root = run_dir / "checkpoints"

    def directory_stage(path: Path) -> int | None:
        name = path.parent.name
        match = re.fullmatch(r"stage_([0-9]{3})", name)
        if match:
            return int(match.group(1))
        if name.startswith("stage_"):
            raise ValueError(f"Invalid checkpoint stage directory: {path.parent}.")
        return None

    def inspected_selection(path: Path, stage_hint: int | None = None) -> CheckpointSelection:
        info = inspect_checkpoint_metadata(path)
        stage = info.stage_index
        if not isinstance(stage, int) or isinstance(stage, bool) or not 0 <= stage < stage_count:
            raise ValueError(f"Invalid checkpoint stage_index {stage!r} in {path}.")
        if stage_hint is not None and stage != stage_hint:
            raise ValueError(
                f"Checkpoint stage {stage} does not match directory stage {stage_hint}: {path}."
            )
        return CheckpointSelection(path, stage, info.stage_completed)

    # Use the explicitly selected checkpoint.
    if selected is not None:
        value = Path(selected)
        if value.is_absolute():
            path = value
        elif value.parts and value.parts[0] == "checkpoints":
            path = run_dir / value
        else:
            path = root / value
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checkpoint must be inside the historical run.")
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint is missing: {path}.")
        if not path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {path}.")
        return inspected_selection(path, directory_stage(path))

    # Prefer latest.pt from the highest stage directory.
    candidates = []
    for path in root.glob("stage_*/latest.pt"):
        stage = directory_stage(path)
        if stage is not None:
            candidates.append((stage, path))
    candidates.sort()
    if candidates:
        stage, path = candidates[-1]
        return inspected_selection(path, stage)

    # Fall back to the legacy run-level latest.pt.
    legacy = root / "latest.pt"
    if legacy.is_file():
        return inspected_selection(legacy)

    # Fall back to the checkpoint copied when the run was forked.
    archived = root / "resume.pt"
    if archived.is_file():
        return inspected_selection(archived)

    raise FileNotFoundError(f"No checkpoint found under {run_dir}.")
