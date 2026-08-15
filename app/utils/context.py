from dataclasses import dataclass
from typing import Any
from pathlib import Path

import torch

from app.utils.device import resolve_device
from app.utils.seed import set_seed


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    dtype: torch.dtype
    num_threads: int
    seed: int
    deterministic: bool
    load_dir: Path
    save_dir: Path


def create_runtime_context(
    runtime_config: dict[str, Any],
    load_dir: Path,
    save_dir: Path,
) -> RuntimeContext:
    """setup runtime context according to config."""

    if runtime_config is None:
        raise ValueError("Missing runtime configuration.")

    if not runtime_config:
        raise ValueError("Runtime configuration is Empty.")

    device = resolve_device(str(runtime_config.get("device")))

    dtype_name = str(runtime_config.get("dtype"))
    if dtype_name not in DTYPE_MAP:
        valid_names = ", ".join(DTYPE_MAP)
        raise ValueError(
            f"Invalid data type: {dtype_name!r}. "
            f"Expected one of {valid_names}."
        )
    dtype = DTYPE_MAP[dtype_name]

    num_threads = int(runtime_config.get("num_threads", 0))
    seed = runtime_config.get("seed")
    deterministic = bool(runtime_config.get("deterministic"))


    if num_threads > 0:
        torch.set_num_threads(num_threads)
    
    seed, deterministic = set_seed(
        seed=seed,
        deterministic=deterministic,
    )

    context = RuntimeContext(
            device=device,
            dtype=dtype,
            num_threads=num_threads,
            seed=seed,
            deterministic=deterministic,
            load_dir=load_dir,
            save_dir=save_dir,
        )

    return context