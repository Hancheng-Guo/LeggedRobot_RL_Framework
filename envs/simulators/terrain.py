"""Backend-independent configuration for geometric terrain.

Reference geometry and allocation are shared; collision creation and world
height queries belong to each simulator backend. This module has no simulator
imports and never consumes a process-global random stream.
"""

from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
    difficulty_levels: tuple[float, ...] = (1.0,)
    spawn_size: tuple[float, float] = (0.5, 0.5)


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
            "max_slope", "spawn_margin", "seed", "difficulty_levels", "spawn_size",
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
        levels = tuple(cls._finite_number(item, "difficulty_levels")
                       for item in sequence("difficulty_levels", (1.0,)))
        if not levels or any(not 0 <= level <= 1 for level in levels) or len(set(levels)) != len(levels):
            raise ValueError("terrain.difficulty_levels must be unique values in [0, 1].")
        raw_spawn = sequence("spawn_size", (0.5, 0.5))
        if len(raw_spawn) != 2:
            raise ValueError("terrain.spawn_size must contain two lengths.")
        spawn_size = tuple(cls._finite_number(item, "spawn_size") for item in raw_spawn)
        if any(size <= 0 or size + 2 * spawn_margin >= length
               for size, length in zip(spawn_size, patch_size)):
            raise ValueError("terrain.spawn_size must fit inside the spawn margin.")
        return cls(types, weights, patch_size, resolution, max_height, max_slope,
                   spawn_margin, seed, levels, (spawn_size[0], spawn_size[1]))


    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"terrain.{name} must contain finite numbers.")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"terrain.{name} must contain finite numbers.")
        return result


def _uniform(seed: int, label: str) -> float:
    """Versioned, stateless 53-bit uniform; never consumes a training RNG."""
    digest = hashlib.blake2b(f"terrain-v1:{seed}:{label}".encode("ascii"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") >> 11) / (1 << 53)


@dataclass(frozen=True)
class TerrainPatch:
    """Local metre coordinates, centred at zero; rows=y, columns=x.

    Backends must use the supplied vertices, split each cell on its lower-left
    to upper-right diagonal, and query those same triangles. No extrapolation
    outside the patch is defined. The central spawn rectangle is flat at z=0.
    """

    patch_id: int
    seed: int
    terrain_type: str
    difficulty: float
    slope: float
    x: tuple[float, ...]
    y: tuple[float, ...]
    heights: tuple[tuple[float, ...], ...]
    spawn_size: tuple[float, float]


def generate_terrain_patch(config: TerrainConfig, runtime_seed: int, patch_id: int,
                           difficulty_index: int = 0) -> TerrainPatch:
    """Deterministic reference geometry shared by future backend adapters.

    Resolution is a maximum cell size (ceil(size/resolution) equal cells).
    Rough vertices are independent uniform heights in +/- max_height*difficulty;
    slope runs along x with a random sign and magnitude <= max_slope*difficulty.
    Vertices within one cell of the spawn rectangle are zero, guaranteeing that
    all triangles intersecting that rectangle are flat. Slope resumes outside
    that plateau. max_slope constrains slope terrain only, not rough gradients.
    """
    if isinstance(difficulty_index, bool) or not isinstance(difficulty_index, int) or not 0 <= difficulty_index < len(config.difficulty_levels):
        raise ValueError("Invalid terrain difficulty index.")
    seed = terrain_patch_seed(config.effective_seed(runtime_seed), patch_id)
    # Normalize before summation to avoid overflowing otherwise valid weights.
    weights = [weight / max(config.weights) for weight in config.weights]
    draw = _uniform(seed, "type") * sum(weights)
    terrain_type = config.types[-1]
    for name, weight in zip(config.types, weights):
        draw -= weight
        if draw < 0:
            terrain_type = name
            break
    difficulty = config.difficulty_levels[difficulty_index]
    slope = (2 * _uniform(seed, "slope") - 1) * config.max_slope * difficulty if terrain_type == "slope" else 0.0
    nx, ny = (math.ceil(size / config.resolution) for size in config.patch_size)
    dx, dy = config.patch_size[0] / nx, config.patch_size[1] / ny
    x = tuple(-config.patch_size[0] / 2 + i * dx for i in range(nx + 1))
    y = tuple(-config.patch_size[1] / 2 + j * dy for j in range(ny + 1))
    safe_x, safe_y = config.spawn_size[0] / 2 + dx, config.spawn_size[1] / 2 + dy
    rows = []
    for j, py in enumerate(y):
        row = []
        for i, px in enumerate(x):
            height = 0.0
            if terrain_type == "slope":
                height = slope * math.copysign(max(0.0, abs(px) - safe_x), px)
            elif terrain_type == "rough" and (abs(px) > safe_x or abs(py) > safe_y):
                height = (2 * _uniform(seed, f"height:{j}:{i}") - 1) * config.max_height * difficulty
            row.append(height)
        rows.append(tuple(row))
    return TerrainPatch(patch_id, seed, terrain_type, difficulty, slope, x, y,
                        tuple(rows), config.spawn_size)


class TerrainAssignments:
    """Reference allocation lifecycle, independent of backend and training RNG.

    Construct once before scene creation. Initial generation is zero. Only reset
    selected environments increments their generation; difficulty updates apply
    at that boundary, before spawn placement and task reset. Backends may map
    these logical patches to prebuilt geometry instead of rebuilding collisions.
    """

    def __init__(self, config: TerrainConfig, runtime_seed: int, num_envs: int):
        terrain_patch_seed(runtime_seed, 0)
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer.")
        self.config = config
        self.seed = config.effective_seed(runtime_seed)
        self.num_envs = num_envs
        self.generations = [0] * num_envs
        self.difficulty_indices = [0] * num_envs

    def patch(self, env_id: int) -> TerrainPatch:
        self._validate_ids([env_id])
        return generate_terrain_patch(self.config, self.seed,
                                      self.generations[env_id] * self.num_envs + env_id,
                                      self.difficulty_indices[env_id])

    def _validate_ids(self, env_ids: Sequence[int]) -> None:
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self.num_envs for i in env_ids):
            raise ValueError("Invalid terrain environment IDs.")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("Terrain environment IDs must be unique.")

    def reset(self, env_ids: Sequence[int], difficulty_indices: Sequence[int] | None = None) -> None:
        self._validate_ids(env_ids)
        levels = list(difficulty_indices) if difficulty_indices is not None else [self.difficulty_indices[i] for i in env_ids]
        if len(levels) != len(env_ids) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self.config.difficulty_levels) for i in levels):
            raise ValueError("Invalid terrain difficulty indices.")
        for env_id, level in zip(env_ids, levels):
            self.generations[env_id] += 1
            self.difficulty_indices[env_id] = level

    def state_dict(self) -> dict[str, Any]:
        return dict(version=1, config=asdict(self.config), seed=self.seed,
                    num_envs=self.num_envs, generations=self.generations.copy(),
                    difficulty_indices=self.difficulty_indices.copy())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Validate all fields before mutating live allocation state.
        if not isinstance(state, Mapping) or set(state) != {
            "version", "config", "seed", "num_envs", "generations", "difficulty_indices",
        } or not isinstance(state["config"], Mapping):
            raise ValueError("Invalid terrain allocation checkpoint schema.")
        if state.get("version") != 1 or state.get("num_envs") != self.num_envs or state.get("seed") != self.seed:
            raise ValueError("Incompatible terrain allocation checkpoint.")
        if TerrainConfig.from_mapping(state.get("config")) != self.config:
            raise ValueError("Terrain checkpoint configuration differs.")
        generations = list(state["generations"])
        levels = list(state["difficulty_indices"])
        if len(generations) != self.num_envs or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in generations):
            raise ValueError("Invalid terrain generations.")
        if len(levels) != self.num_envs or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self.config.difficulty_levels) for i in levels):
            raise ValueError("Invalid terrain difficulty indices.")
        self.generations, self.difficulty_indices = generations, levels
