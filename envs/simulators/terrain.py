"""Backend-independent configuration for geometric terrain.

Geometry creation and height sampling belong to each simulator backend.  This
module deliberately has no MuJoCo or Isaac Sim imports.
"""

from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


TERRAIN_TYPES = frozenset({
    "plane",
    "slope",
    "rough",
})


def terrain_patch_seed(
    base_seed: int,
    patch_id: int,
) -> int:
    """Stable 64-bit seed for one patch, independent of generation order."""

    if any(
        (
            isinstance(value, bool) or
            not isinstance(value, int) or
            value < 0
        )
        for value in (base_seed, patch_id)
    ):
        raise ValueError("Terrain seed and patch ID must be non-negative integers.")
    payload = f"terrain-v1:{base_seed}:{patch_id}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class TerrainConfig:
    """Shared terrain parameters; lengths are metres and slopes are rise/run.

    ``types`` and ``weights`` describe the per-environment distribution.  A
    fixed seed overrides the runtime seed; None uses the runtime seed.  The
    current backends only build the default plane; non-plane configurations
    are rejected until their geometry implementations are added.
    """

    types: tuple[str, ...] = ("plane",)
    weights: tuple[float, ...] = (1.0,)
    patch_size: tuple[float, float] = (4.0, 4.0)
    resolution: float = 0.1
    max_height: float = 0.0
    max_slope: float = 0.0
    spawn_margin: float = 0.5
    seed: int | None = None


    @property
    def is_plane(self) -> bool:
        return self.types == ("plane",)


    def effective_seed(
        self,
        runtime_seed: int,
    ) -> int:
        return runtime_seed if self.seed is None else self.seed


    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> TerrainConfig:

        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("'terrain' must be a mapping.")
        unknown = set(value) - {
            "types", "weights", "patch_size", "resolution", "max_height",
            "max_slope", "spawn_margin", "seed",
        }
        if unknown:
            raise ValueError(f"Unknown terrain field(s): {sorted(unknown)}.")

        def sequence(name: str, default: Sequence[Any]) -> Sequence[Any]:
            item = value.get(name, default)
            if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
                raise TypeError(f"terrain.{name} must be a sequence.")
            return item

        types = tuple(sequence("types", ("plane",)))
        if not types or any(not isinstance(terrain_type, str) for terrain_type in types):
            raise ValueError("terrain.types must contain terrain type names.")
        if len(set(types)) != len(types) or set(types) - TERRAIN_TYPES:
            raise ValueError(f"terrain.types must be unique members of {sorted(TERRAIN_TYPES)}.")

        raw_weights = sequence("weights", (1.0,) * len(types))
        if len(raw_weights) != len(types):
            raise ValueError("terrain.weights must match terrain.types in length.")
        weights = tuple(cls._finite_number(item, "weights") for item in raw_weights)
        if any(weight <= 0 for weight in weights):
            raise ValueError("terrain.weights must be positive.")

        raw_size = sequence("patch_size", (4.0, 4.0))
        if len(raw_size) != 2:
            raise ValueError("terrain.patch_size must contain two lengths.")
        patch_size = (
            cls._finite_number(raw_size[0], "patch_size"),
            cls._finite_number(raw_size[1], "patch_size"),
        )
        resolution = cls._finite_number(value.get("resolution", 0.1), "resolution")
        max_height = cls._finite_number(value.get("max_height", 0.0), "max_height")
        max_slope = cls._finite_number(value.get("max_slope", 0.0), "max_slope")
        spawn_margin = cls._finite_number(value.get("spawn_margin", 0.5), "spawn_margin")
        if any(length <= 0 for length in patch_size) or resolution <= 0:
            raise ValueError("terrain.patch_size and terrain.resolution must be positive.")
        if min(patch_size) < 2 * resolution:
            raise ValueError("terrain.patch_size must span at least two resolution cells.")
        if max_height < 0 or max_slope < 0 or spawn_margin < 0:
            raise ValueError("terrain height, slope, and spawn margin must be non-negative.")
        if 2 * spawn_margin >= min(patch_size):
            raise ValueError("terrain.spawn_margin leaves no interior spawn area.")
        if "rough" in types and max_height == 0:
            raise ValueError("terrain.max_height must be positive for rough terrain.")
        if "slope" in types and max_slope == 0:
            raise ValueError("terrain.max_slope must be positive for slope terrain.")

        seed = value.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("terrain.seed must be a non-negative integer or null.")
        return cls(types, weights, patch_size, resolution, max_height, max_slope, spawn_margin, seed)


    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"terrain.{name} must contain finite numbers.")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"terrain.{name} must contain finite numbers.")
        return result
