from pathlib import Path
from typing import Any

import torch


def atomic_save(
    payload: dict[str, Any],
    path: Path,
) -> Path:
    """Atomically save a PyTorch payload to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path
