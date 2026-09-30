"""Runtime settings shared by the application and its components."""

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    dtype: torch.dtype
    num_threads: int
    seed: int
    deterministic_ops: bool
    load_dir: Path
    save_dir: Path
