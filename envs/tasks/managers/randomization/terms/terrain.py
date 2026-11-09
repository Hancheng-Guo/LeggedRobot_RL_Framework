"""Task terrain randomization: configuration, generation, allocation and spawn.

Physical operations go through the backend-neutral contract in simulators.terrain;
this module never imports MuJoCo or Isaac Sim APIs.
"""
from __future__ import annotations

import math
import hashlib
import torch
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from copy import deepcopy
from typing import cast, Any, TypeAlias

from envs.simulators.terrain import TerrainBackend, TerrainGeometry, TerrainPatch
from utils import RuntimeContext
from .base import BaseRandomizationTerm, RandomizationBackend
from .registry import register_randomization


@dataclass(frozen=True)
class TerrainAssignment:
    """Task policy metadata plus a backend-only geometry payload."""
    patch_id: int
    seed: int
    terrain_type: str
    difficulty: float
    geometry: TerrainPatch
    spawn_size: tuple[float, float]


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
class PlaneTerrainConfig:
    weight: float = 1.0


@dataclass(frozen=True)
class SlopeTerrainConfig:
    max_slope: float
    weight: float = 1.0


@dataclass(frozen=True)
class RoughTerrainConfig:
    max_height: float
    weight: float = 1.0


TerrainTypeConfig: TypeAlias = PlaneTerrainConfig | SlopeTerrainConfig | RoughTerrainConfig


@dataclass(frozen=True)
class TerrainConfig:
    """Shared terrain parameters; lengths are metres and slopes are rise/run.

    Each type owns its weight and specific parameters. A
    fixed seed overrides the runtime seed; None uses the runtime seed. MuJoCo
    builds these grids as heightfields. Isaac Sim still rejects
    non-plane configurations until its geometry implementation is added.
    """

    types: dict[str, TerrainTypeConfig] = field(
        default_factory=lambda: {"plane": PlaneTerrainConfig()})
    patch_size: tuple[float, float] = (4.0, 4.0)
    resolution: float = 0.1
    spawn_margin: float = 0.5
    seed: int | None = None
    difficulty_levels: tuple[float, ...] = (1.0,)
    spawn_size: tuple[float, float] = (0.5, 0.5)


    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(other, TerrainConfig):
            return NotImplemented
        # Type order defines weighted sampling intervals and must match on restore.
        return (
            tuple(self.types) == tuple(other.types)
            and asdict(self) == asdict(other)
        )


    @property
    def is_plane(self) -> bool:
        return len(self.types) == 1 and "plane" in self.types


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
            "types", "patch_size", "resolution", "spawn_margin",
            "seed", "difficulty_levels", "spawn_size",
        }
        if unknown:
            raise ValueError(f"Unknown terrain field(s): {sorted(unknown)}.")

        def sequence(name: str, default: Sequence[Any]) -> Sequence[Any]:
            item = value.get(name, default)
            if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
                raise TypeError(f"terrain.{name} must be a sequence.")
            return item

        raw_types = value.get("types", {"plane": {}})
        if not isinstance(raw_types, Mapping):
            raise TypeError(
                "terrain.types must be a mapping keyed by terrain type names."
            )
        if not raw_types:
            raise ValueError("terrain.types must contain at least one terrain type.")

        types: dict[str, TerrainTypeConfig] = {}
        for name, params in raw_types.items():
            if not isinstance(params, Mapping):
                raise TypeError(f"terrain.types.{name} must be a mapping.")

            weight = cls._finite_number(params.get("weight", 1.0), f"types.{name}.weight")
            if weight <= 0:
                raise ValueError(f"terrain.types.{name}.weight must be positive.")
            allowed = {"weight"}
            if name == "slope":
                allowed.add("max_slope")
            elif name == "rough":
                allowed.add("max_height")
            if set(params) - allowed:
                raise ValueError(
                    f"Unknown terrain.types.{name} field(s): {sorted(set(params) - allowed)}."
                )

            if name == "slope":
                max_slope = cls._finite_number(params.get("max_slope", 0.0), "types.slope.max_slope")
                if max_slope <= 0:
                    raise ValueError("terrain.types.slope.max_slope must be positive.")
                types[name] = SlopeTerrainConfig(max_slope=max_slope, weight=weight)
            elif name == "rough":
                max_height = cls._finite_number(params.get("max_height", 0.0), "types.rough.max_height")
                if max_height <= 0:
                    raise ValueError("terrain.types.rough.max_height must be positive.")
                types[name] = RoughTerrainConfig(max_height=max_height, weight=weight)
            elif name == "plane":
                types[name] = PlaneTerrainConfig(weight=weight)
            else:
                raise ValueError(f"Unsupported terrain type: {name}.")

        raw_size = sequence("patch_size", (4.0, 4.0))
        if len(raw_size) != 2:
            raise ValueError("terrain.patch_size must contain two lengths.")
        patch_size = (
            cls._finite_number(raw_size[0], "patch_size"),
            cls._finite_number(raw_size[1], "patch_size"),
        )
        resolution = cls._finite_number(value.get("resolution", 0.1), "resolution")
        spawn_margin = cls._finite_number(value.get("spawn_margin", 0.5), "spawn_margin")
        if any(length <= 0 for length in patch_size) or resolution <= 0:
            raise ValueError("terrain.patch_size and terrain.resolution must be positive.")
        if min(patch_size) < 2 * resolution:
            raise ValueError("terrain.patch_size must span at least two resolution cells.")
        if spawn_margin < 0:
            raise ValueError("terrain.spawn_margin must be non-negative.")
        if 2 * spawn_margin >= min(patch_size):
            raise ValueError("terrain.spawn_margin leaves no interior spawn area.")

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
        return cls(
            types=types,
            patch_size=patch_size,
            resolution=resolution,
            spawn_margin=spawn_margin,
            seed=seed,
            difficulty_levels=levels,
            spawn_size=(spawn_size[0], spawn_size[1]),
        )


    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"terrain.{name} must contain finite numbers.")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"terrain.{name} must contain finite numbers.")
        return result


def _uniform(
    seed: int,
    label: str,
) -> float:
    """Versioned, stateless 53-bit uniform; never consumes a training RNG."""
    digest = hashlib.blake2b(f"terrain-v1:{seed}:{label}".encode("ascii"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") >> 11) / (1 << 53)


def generate_terrain_patch(
    config: TerrainConfig,
    runtime_seed: int,
    patch_id: int,
    difficulty_index: int = 0,
) -> TerrainAssignment:
    """Deterministic reference geometry shared by future backend adapters.

    Resolution is a maximum cell size (ceil(size/resolution) equal cells).
    Rough vertices are independent uniform heights in +/- max_height*difficulty;
    slope rises in all four directions outside the spawn area when positive,
    and falls in all four directions when negative.
    Vertices within one cell of the spawn rectangle are zero, guaranteeing that
    all triangles intersecting that rectangle are flat. Slope resumes outside
    that plateau and returns to zero at the patch perimeter to meet the exterior
    flat ground. max_slope bounds the sampled coefficient, not the total gradient
    magnitude of collision triangles, which can combine x and y gradients.
    """
    if (
        isinstance(difficulty_index, bool)
        or not isinstance(difficulty_index, int)
        or not 0 <= difficulty_index < len(config.difficulty_levels)
    ):
        raise ValueError("Invalid terrain difficulty index.")
    seed = terrain_patch_seed(config.effective_seed(runtime_seed), patch_id)

    # Normalize before summation to avoid overflowing otherwise valid weights.
    max_weight = max(params.weight for params in config.types.values())
    weights = [params.weight / max_weight for params in config.types.values()]
    draw = _uniform(seed, "type") * sum(weights)
    terrain_type = next(reversed(config.types))
    for name, weight in zip(config.types, weights):
        draw -= weight
        if draw < 0:
            terrain_type = name
            break
    difficulty = config.difficulty_levels[difficulty_index]
    nx, ny = (math.ceil(size / config.resolution) for size in config.patch_size)
    dx, dy = config.patch_size[0] / nx, config.patch_size[1] / ny
    x = tuple(-config.patch_size[0] / 2 + i * dx for i in range(nx + 1))
    y = tuple(-config.patch_size[1] / 2 + j * dy for j in range(ny + 1))
    safe_x, safe_y = config.spawn_size[0] / 2 + dx, config.spawn_size[1] / 2 + dy

    rows: list[tuple[float, ...]] = []
    type_config = config.types[terrain_type]

    if isinstance(type_config, SlopeTerrainConfig):
        slope = (2 * _uniform(seed, "slope") - 1) * type_config.max_slope * difficulty
        for py in y:
            row = []
            for px in x:
                distance_from_spawn = max(abs(px) - safe_x, abs(py) - safe_y)
                distance_to_x_edge = config.patch_size[0] / 2 - abs(px)
                distance_to_y_edge = config.patch_size[1] / 2 - abs(py)
                run = max(
                    0.,
                    min(distance_from_spawn, distance_to_x_edge, distance_to_y_edge)
                )
                row.append(slope * run)
            rows.append(tuple(row))

    elif isinstance(type_config, RoughTerrainConfig):
        for j, py in enumerate(y):
            row = []
            for i, px in enumerate(x):
                height = 0.0
                if (i not in (0, nx) and j not in (0, ny)
                        and (abs(px) > safe_x or abs(py) > safe_y)):
                    height = (
                        (2 * _uniform(seed, f"height:{j}:{i}") - 1)
                        * type_config.max_height * difficulty
                    )
                row.append(height)
            rows.append(tuple(row))

    elif isinstance(type_config, PlaneTerrainConfig):
        rows = [tuple(0.0 for _ in x) for _ in y]

    else:
        raise TypeError(f"Unsupported terrain configuration type: {type(type_config).__name__}.")

    return TerrainAssignment(
        patch_id=patch_id,
        seed=seed,
        terrain_type=terrain_type,
        difficulty=difficulty,
        geometry=TerrainPatch(x, y, tuple(rows)),
        spawn_size=config.spawn_size,
    )


class TerrainAssignments:
    """Reference allocation lifecycle, independent of backend and training RNG.

    Initial generation is zero. Explicit allocation reset advances selected
    patches; episode resets can reuse them without advancing generations.
    Difficulty updates apply before spawn placement. Backends may map
    these logical patches to prebuilt geometry instead of rebuilding collisions.
    """

    def __init__(
        self,
        config: TerrainConfig,
        runtime_seed: int,
        num_envs: int,
    ):
        if (
            isinstance(runtime_seed, bool)
            or not isinstance(runtime_seed, int)
            or runtime_seed < 0
        ):
            raise ValueError("runtime_seed must be a non-negative integer.")
        if (
            isinstance(num_envs, bool)
            or not isinstance(num_envs, int)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer.")

        self.config = config
        self.seed = config.effective_seed(runtime_seed)
        self.num_envs = num_envs
        self.generations = [0] * num_envs
        self.difficulty_indices = [0] * num_envs


    def patch(
        self,
        env_id: int,
    ) -> TerrainAssignment:

        self._validate_ids([env_id])
        return self._patch(env_id)


    def _patch(
        self,
        env_id: int,
    ) -> TerrainAssignment:
        """Generate a patch for an already validated environment ID."""

        return generate_terrain_patch(self.config,
                                      self.seed,
                                      self.generations[env_id] * self.num_envs + env_id,
                                      self.difficulty_indices[env_id])


    def _validate_ids(
        self,
        env_ids_list: Sequence[int],
    ) -> None:

        if any(
            isinstance(env_id, bool)
            or not isinstance(env_id, int)
            or not 0 <= env_id < self.num_envs
            for env_id in env_ids_list
        ):
            raise ValueError("Invalid terrain environment IDs.")
        if len(set(env_ids_list)) != len(env_ids_list):
            raise ValueError("Terrain environment IDs must be unique.")


    def reset(
        self,
        env_ids_list: Sequence[int],
        difficulty_indices: Sequence[int] | None = None,
    ) -> None:

        self._validate_ids(env_ids_list)
        self._reset(env_ids_list, difficulty_indices)


    def _reset(
        self,
        env_ids_list: Sequence[int],
        difficulty_indices: Sequence[int] | None,
    ) -> None:
        """Advance validated environment IDs, validating only the new difficulties."""

        levels = (
            list(difficulty_indices)
            if difficulty_indices is not None
            else [self.difficulty_indices[env_id] for env_id in env_ids_list]
        )
        if (
            len(levels) != len(env_ids_list)
            or any(
                isinstance(env_id, bool)
                or not isinstance(env_id, int)
                or not 0 <= env_id < len(self.config.difficulty_levels)
                for env_id in levels
            )
        ):
            raise ValueError("Invalid terrain difficulty indices.")
        for env_id, level in zip(env_ids_list, levels):
            self.generations[env_id] += 1
            self.difficulty_indices[env_id] = level


    def state_dict(self) -> dict[str, Any]:
        return dict(version=5,
                    config=asdict(self.config),
                    seed=self.seed,
                    num_envs=self.num_envs,
                    generations=self.generations.copy(),
                    difficulty_indices=self.difficulty_indices.copy())


    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        # Validate all fields before mutating live allocation state.

        state_set = {
            "version", "config", "seed", "num_envs",
            "generations", "difficulty_indices",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != state_set
            or not isinstance(state["config"], Mapping)
        ):
            raise ValueError("Invalid terrain allocation checkpoint schema.")
        if (
            state.get("version") != 5
            or state.get("num_envs") != self.num_envs
            or state.get("seed") != self.seed
        ):
            raise ValueError("Incompatible terrain allocation checkpoint.")
        if TerrainConfig.from_mapping(state.get("config")) != self.config:
            raise ValueError("Terrain checkpoint configuration differs.")
        generations = list(state["generations"])
        levels = list(state["difficulty_indices"])
        if (
            len(generations) != self.num_envs
            or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or i < 0
                for i in generations
            )
        ):
            raise ValueError("Invalid terrain generations.")
        if (
            len(levels) != self.num_envs
            or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < len(self.config.difficulty_levels)
                for i in levels  
            )
        ):
            raise ValueError("Invalid terrain difficulty indices.")

        self.generations = generations
        self.difficulty_indices = levels


@dataclass
class _PendingTerrainReset:
    env_ids_tuple: tuple[int, ...]
    candidate: TerrainAssignments
    patches: tuple[TerrainAssignment, ...]
    physical_state: object


@register_randomization
class TerrainRandomizationTerm(BaseRandomizationTerm):
    """Own terrain choices under the task's RandomizationManager.

    Bind after simulator configuration and before episode resets. before_simulator_reset
    supplies geometry; the environment resets physics; after_simulator_reset supplies the
    spawn pose. The term never owns the physical reset operation.
    """
    def __init__(
        self,
        num_envs: int,
        context: RuntimeContext,
        **params,
    ):

        super().__init__(num_envs, context, **params)

        self.config: TerrainConfig
        self.assignments = None if self.config.is_plane else TerrainAssignments(self.config, context.seed, num_envs)
        self.geometry = self.geometry_from_config(self.config)
        self.simulator: TerrainBackend
        self._patches: list[TerrainAssignment | None] = [None] * num_envs if self.assignments is not None else []
        self._pending: _PendingTerrainReset | None = None
        self._requested_difficulties: dict[int, int] = {}


    @classmethod
    def normalize_config(
        cls,
        params: Mapping[str, Any],
    ) -> TerrainConfig:
        return TerrainConfig.from_mapping(params)


    @classmethod
    def get_simulator_requirements(
        cls,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        geometry = cls.geometry_from_config(cls.normalize_config(params))
        return {} if geometry is None else {"terrain_geometry": geometry}


    @staticmethod
    def geometry_from_config(config: TerrainConfig) -> TerrainGeometry | None:
        """Declare collision buffers without constructing allocation state."""
        if config.is_plane:
            return None
        height_bound = 0.0
        for params in config.types.values():
            if isinstance(params, RoughTerrainConfig):
                height_bound = max(height_bound, params.max_height)
            elif isinstance(params, SlopeTerrainConfig):
                height_bound = max(height_bound, params.max_slope * config.patch_size[0] / 2)
        return TerrainGeometry(
            config.patch_size,
            (
                math.ceil(config.patch_size[1] / config.resolution) + 1,
                math.ceil(config.patch_size[0] / config.resolution) + 1
            ),
            height_bound,
        )

    @property
    def patches(self) -> tuple[TerrainAssignment | None, ...]:
        return tuple(self._patches)


    def bind(
        self,
        backend: RandomizationBackend,
    ) -> None:

        simulator = cast(TerrainBackend, backend)
        if simulator.num_envs != self.num_envs:
            raise ValueError("Terrain term and simulator environment counts differ.")

        if self.assignments is not None:
            measured = simulator.get_terrain_spawn_geometry()
            half = [size / 2 for size in self.config.spawn_size]
            lower = [-half[i] - measured.collision_lower[i] for i in range(2)]
            upper = [half[i] - measured.collision_upper[i] for i in range(2)]
            if any(a > b for a, b in zip(lower, upper)):
                footprint = [measured.collision_upper[i] - measured.collision_lower[i] for i in range(2)]
                raise ValueError(
                    f"terrain.spawn_size cannot contain the robot collision footprint {footprint}; "
                    "enlarge the safe spawn area."
                )
            self._spawn_bounds = (lower[0], upper[0], lower[1], upper[1])
            self._spawn_geometry = measured
            self._spawn_lift = max(0., measured.origin[2] + .002
                                   - measured.root_position[2] - measured.collision_lower[2])
        self.simulator = simulator
        # A newly configured backend needs the current geometry installed once.
        self._unapplied_envs = set(range(self.num_envs))


    def _selected_ids(
        self,
        env_ids: torch.Tensor | None,
    ) -> list[int]:

        if env_ids is None:
            return list(range(self.num_envs))
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError("Environment IDs must be a one-dimensional long tensor.")
        env_ids_list = env_ids.tolist()
        if len(set(env_ids_list)) != len(env_ids_list) or any(env_id < 0 or env_id >= self.num_envs for env_id in env_ids_list):
            raise ValueError("Environment IDs must be unique and in range.")
        return env_ids_list


    def before_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
        difficulty_indices: torch.Tensor | None = None,
    ) -> None:
        """Reuse terrain unless uninitialized or its requested difficulty changes."""

        env_ids_list = self._selected_ids(env_ids)
        if (
            difficulty_indices is not None
            and (difficulty_indices.ndim != 1 or difficulty_indices.dtype != torch.long)
        ):
            raise ValueError(
                "Terrain difficulty indices must be a one-dimensional long tensor."
            )
        if self.assignments is None:
            if difficulty_indices is not None:
                raise ValueError("Difficulty updates require geometric terrain.")
            return
        if not env_ids_list:
            return
        if self._pending is not None:
            raise RuntimeError("Finish or abort the previous terrain reset first.")

        candidate = deepcopy(self.assignments)
        levels = (
            difficulty_indices.tolist()
            if difficulty_indices is not None
            else [self._requested_difficulties.get(env_id, candidate.difficulty_indices[env_id]) for env_id in env_ids_list])
        if len(levels) != len(env_ids_list):
            raise ValueError("Terrain difficulty indices must match selected environments.")

        changed = {env_id for env_id, level in zip(env_ids_list, levels)
                   if self._patches[env_id] is None or level != candidate.difficulty_indices[env_id]}
        candidate._reset([env_id for env_id in env_ids_list if env_id in changed],
                        [level for env_id, level in zip(env_ids_list, levels) if env_id in changed])
        patches = tuple(candidate._patch(env_id) if env_id in changed else cast(TerrainAssignment, self._patches[env_id])
                        for env_id in env_ids_list)
        env_ids_tensor = torch.tensor(env_ids_list, dtype=torch.long, device=self.simulator.context.device)
        snapshot = self.simulator.capture_reset_state(env_ids_tensor)
        self._pending = _PendingTerrainReset(tuple(env_ids_list), candidate, patches, snapshot)

        try:
            updates = [(env_id, patch.geometry) for env_id, patch in zip(env_ids_list, patches)
                       if env_id in changed or env_id in self._unapplied_envs]
            if updates:
                self.simulator.apply_terrain(
                    torch.tensor([env_id for env_id, _ in updates], dtype=torch.long, device=env_ids_tensor.device),
                    [geometry for _, geometry in updates],
                )
        except Exception as error:
            try:
                self.abort_simulator_reset(env_ids)
            except Exception as rollback_error:
                raise ExceptionGroup(
                    "Terrain update and rollback failed",
                    [error, rollback_error]
                ) from error
            raise


    def after_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Apply spawn positions after the environment has reset physics."""

        env_ids_list = self._selected_ids(env_ids)
        if self.assignments is None or not env_ids_list:
            return
        if self._pending is None or self._pending.env_ids_tuple != tuple(env_ids_list):
            raise RuntimeError("Terrain post-reset must match a pending pre-reset.")
        env_ids_tensor = torch.tensor(env_ids_list, dtype=torch.long, device=self.simulator.context.device)
        try:
            positions = self._spawn_positions(env_ids_tensor, self._pending.patches)
            self.simulator.set_root_positions(env_ids_tensor, positions)
        except Exception as error:
            try:
                self.abort_simulator_reset(env_ids)
            except Exception as rollback_error:
                raise ExceptionGroup(
                    "Terrain spawn and rollback failed",
                    [error, rollback_error]
                ) from error
            raise


    def commit_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if self._pending is None:
            return
        assert self.assignments is not None
        pending = self._pending
        self.assignments.load_state_dict(pending.candidate.state_dict())
        for env_id, patch in zip(pending.env_ids_tuple, pending.patches):
            self._patches[env_id] = patch
            self._requested_difficulties.pop(env_id, None)
            self._unapplied_envs.discard(env_id)
        self._pending = None


    def abort_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if self._pending is None:
            return
        pending = self._pending
        self._pending = None
        self.simulator.restore_reset_state(pending.physical_state)


    def set_difficulty(
        self,
        env_ids_tensor: torch.Tensor,
        difficulty_indices: torch.Tensor,
    ) -> None:

        if self.assignments is None:
            raise ValueError("Difficulty updates require geometric terrain.")
        env_ids_list = self._selected_ids(env_ids_tensor)
        if (
            difficulty_indices.ndim != 1
            or difficulty_indices.dtype != torch.long
            or len(env_ids_list) != difficulty_indices.numel()
            or any(not 0 <= i < len(self.config.difficulty_levels) for i in difficulty_indices.tolist())
        ):
            raise ValueError("Invalid terrain difficulty indices.")
        self._requested_difficulties.update(zip(env_ids_list, difficulty_indices.tolist()))


    def _spawn_positions(
        self,
        env_ids_tensor: torch.Tensor,
        patches: Sequence[TerrainAssignment],
    ) -> torch.Tensor:

        measured = self._spawn_geometry
        xmin, xmax, ymin, ymax = self._spawn_bounds
        positions = torch.tensor([
            [measured.origin[0] + xmin + (xmax - xmin) * _uniform(p.seed, "spawn:x"),
             measured.origin[1] + ymin + (ymax - ymin) * _uniform(p.seed, "spawn:y"),
             measured.root_position[2] + self._spawn_lift - measured.origin[2]]
            for p in patches
        ], dtype=torch.float64, device=self.simulator.context.device)
        positions[:, 2] += self.simulator.get_ground_heights(positions[:, None, :2], env_ids_tensor)[:, 0]
        return positions
