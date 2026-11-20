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
from typing import cast, Any

from .base import BaseRandomizationTerm, RandomizationBackend
from .registry import register_randomization
from .terrain_category import (
    TerrainCategoryConfig, TerrainCategoryTerm, PlaneTerrainConfig,
    SlopeTerrainConfig, RoughTerrainConfig, finite_number,
)
from .terrain_difficulty import (
    TerrainDifficultyConfig, TerrainDifficultyTerm, difficulty_value
)
from envs.simulators.terrain import (
    StaticTerrainBackend, TerrainBackend, TerrainGeometry, TerrainPatch, TerrainRegionKey,
)
from envs.tasks.managers.curriculum import (
    CurriculumTermProvider, CurriculumSpaceConfig, CurriculumSpaceConfigs,
)
from utils import RuntimeContext


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
class TerrainConfig:
    """Shared terrain parameters; lengths are metres and slopes are rise/run.

    Each type owns its weight and specific parameters. A
    fixed seed overrides the runtime seed; None uses the runtime seed. MuJoCo
    builds these grids as heightfields. Isaac Sim still rejects
    non-plane configurations until its geometry implementation is added.
    """

    category: TerrainCategoryConfig = field(default_factory=TerrainCategoryConfig)
    difficulty: TerrainDifficultyConfig = field(default_factory=TerrainDifficultyConfig)
    resample_patch: bool = False
    resample_spawn: bool = False
    patch_size: tuple[float, float] = (4.0, 4.0)
    resolution: float = 0.1
    spawn_margin: float = 0.5
    seed: int | None = None
    spawn_size: tuple[float, float] = (0.5, 0.5)

    _finite_number = staticmethod(finite_number)


    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(other, TerrainConfig):
            return NotImplemented
        # Type order defines weighted sampling intervals and must match on restore.
        return (
            self.category == other.category
            and asdict(self) == asdict(other)
        )


    @property
    def is_plane(self) -> bool:
        return self.category.is_plane


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
            "patch_size", "resolution", "spawn_margin",
            "seed", "spawn_size", "category", "difficulty",
            "resample_patch", "resample_spawn",
        }
        if unknown:
            raise ValueError(f"Unknown terrain field(s): {sorted(unknown)}.")

        def sequence(name: str, default: Sequence[Any]) -> Sequence[Any]:
            item = value.get(name, default)
            if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
                raise TypeError(f"terrain.{name} must be a sequence.")
            return item

        category = TerrainCategoryConfig.from_mapping(value.get("category"))
        difficulty = TerrainDifficultyConfig.from_mapping(value.get("difficulty"))
        flags = {name: value.get(name, False) for name in ("resample_patch", "resample_spawn")}
        if any(type(flag) is not bool for flag in flags.values()):
            raise TypeError("resample_patch and resample_spawn must be booleans.")

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
        raw_spawn = sequence("spawn_size", (0.5, 0.5))
        if len(raw_spawn) != 2:
            raise ValueError("terrain.spawn_size must contain two lengths.")
        spawn_size = tuple(cls._finite_number(item, "spawn_size") for item in raw_spawn)
        if any(size <= 0 or size + 2 * spawn_margin >= length
               for size, length in zip(spawn_size, patch_size)):
            raise ValueError("terrain.spawn_size must fit inside the spawn margin.")
        return cls(
            category=category,
            difficulty=difficulty,
            **flags,
            patch_size=patch_size,
            resolution=resolution,
            spawn_margin=spawn_margin,
            seed=seed,
            spawn_size=(spawn_size[0], spawn_size[1]),
        )


    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["category"] = self.category.to_mapping()
        result["difficulty"] = self.difficulty.to_mapping()
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
    terrain_type: str,
    difficulty: float = 1.0,
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
    difficulty = difficulty_value(difficulty)
    seed = terrain_patch_seed(config.effective_seed(runtime_seed), patch_id)

    nx, ny = (math.ceil(size / config.resolution) for size in config.patch_size)
    dx, dy = config.patch_size[0] / nx, config.patch_size[1] / ny
    x = tuple(-config.patch_size[0] / 2 + i * dx for i in range(nx + 1))
    y = tuple(-config.patch_size[1] / 2 + j * dy for j in range(ny + 1))
    safe_x, safe_y = config.spawn_size[0] / 2 + dx, config.spawn_size[1] / 2 + dy

    rows: list[tuple[float, ...]] = []
    type_config = config.category.parameters_for(terrain_type)

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


@dataclass(frozen=True)
class _TerrainCatalog:
    config: TerrainConfig

    @property
    def conditions(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (category, difficulty)
            for category in self.config.category.values
            for difficulty in self.config.difficulty.values
        )

    def patch(self, runtime_seed: int, key: TerrainRegionKey) -> TerrainPatch:
        category, difficulty, variant = key
        return generate_terrain_patch(
            self.config, runtime_seed, variant, category, difficulty,
        ).geometry


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
        self.difficulties = [config.difficulty.initial_value] * num_envs
        self.categories = [config.category.initial_value] * num_envs
        self.allocation_ids = list(range(num_envs))
        self.allocation_stride = num_envs


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

        return generate_terrain_patch(
            self.config,
            self.seed,
            self.generations[env_id] * self.allocation_stride + self.allocation_ids[env_id],
            terrain_type=self.categories[env_id],
            difficulty=self.difficulties[env_id],
        )


    def playback_state_dict(
        self,
        env_index: int,
    ) -> dict[str, Any]:
        """Retain original patch identities when projecting allocation state."""

        self._validate_ids([env_index])
        state = self.state_dict()
        state["num_envs"] = 1
        for name in ("generations", "difficulties", "categories", "allocation_ids"):
            state[name] = [state[name][env_index]]
        return state


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
        difficulties: Sequence[float] | None = None,
        categories: Sequence[str] | None = None,
    ) -> None:

        self._validate_ids(env_ids_list)
        self._reset(env_ids_list, difficulties, categories)


    def _reset(
        self,
        env_ids_list: Sequence[int],
        difficulties: Sequence[float] | None,
        categories: Sequence[str] | None = None,
    ) -> None:
        """Advance validated environment IDs with resolved domain values."""

        selected_difficulties = (
            [self.difficulties[env_id] for env_id in env_ids_list]
            if difficulties is None else list(difficulties)
        )
        selected_categories = (
            [self.categories[env_id] for env_id in env_ids_list]
            if categories is None else list(categories)
        )
        if (
            len(selected_difficulties) != len(env_ids_list)
            or len(selected_categories) != len(env_ids_list)
        ):
            raise ValueError(
                "Terrain selections must match selected environments."
            )

        selected_difficulties = [
            self.config.difficulty.validate_selection(value)
            for value in selected_difficulties
        ]
        selected_categories = [
            self.config.category.validate_selection(value)
            for value in selected_categories
        ]
        for env_id, difficulty, category in (
            zip(env_ids_list, selected_difficulties, selected_categories)
        ):
            self.generations[env_id] += 1
            self.difficulties[env_id] = difficulty
            self.categories[env_id] = category


    def state_dict(self) -> dict[str, Any]:
        return dict(
            version=8,
            config=self.config.to_mapping(),
            seed=self.seed,
            num_envs=self.num_envs,
            generations=self.generations.copy(),
            difficulties=self.difficulties.copy(),
            categories=self.categories.copy(),
            allocation_ids=self.allocation_ids.copy(),
            allocation_stride=self.allocation_stride,
        )


    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        # Validate all fields before mutating live allocation state.

        state_set = {
            "version", "config", "seed", "num_envs",
            "generations", "difficulties", "categories",
            "allocation_ids", "allocation_stride",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != state_set
            or not isinstance(state["config"], Mapping)
        ):
            raise ValueError("Invalid terrain allocation checkpoint schema.")
        if (
            state.get("version") != 8
            or state.get("num_envs") != self.num_envs
            or state.get("seed") != self.seed
        ):
            raise ValueError("Incompatible terrain allocation checkpoint.")
        if TerrainConfig.from_mapping(state.get("config")) != self.config:
            raise ValueError("Terrain checkpoint configuration differs.")
        generations = state["generations"]
        if (
            not isinstance(generations, list)
            or len(generations) != self.num_envs
            or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or i < 0
                for i in generations
            )
        ):
            raise ValueError("Invalid terrain generations.")

        difficulties = state["difficulties"]
        categories = state["categories"]
        if (
            not isinstance(difficulties, list)
            or len(difficulties) != self.num_envs
            or not isinstance(categories, list)
            or len(categories) != self.num_envs
        ):
            raise ValueError("Invalid terrain allocation selections.")
        difficulties = [
            self.config.difficulty.validate_selection(value)
            for value in difficulties
        ]
        categories = [
            self.config.category.validate_selection(value)
            for value in categories
        ]

        allocation_ids = state["allocation_ids"]
        stride = state["allocation_stride"]
        if (
            type(stride) is not int
            or stride <= 0
            or not isinstance(allocation_ids, list)
            or len(allocation_ids) != self.num_envs
            or any(type(i) is not int or not 0 <= i < stride for i in allocation_ids)
            or len(set(allocation_ids)) != self.num_envs
        ):
            raise ValueError("Invalid terrain allocation identity mapping.")

        self.allocation_ids = allocation_ids.copy()
        self.allocation_stride = stride
        self.generations = generations.copy()
        self.difficulties = difficulties
        self.categories = categories


@dataclass
class _PendingTerrainReset:
    env_ids_tuple: tuple[int, ...]
    candidate: TerrainAssignments
    patches: tuple[TerrainAssignment, ...]
    physical_state: object
    spawn_generations: list[int]
    restored_envs: frozenset[int]


@register_randomization
class TerrainRandomizationTerm(BaseRandomizationTerm):
    """Own terrain choices under the task's RandomizationManager.

    Bind after simulator configuration and before episode resets. before_simulator_reset
    supplies geometry; the environment resets physics; after_simulator_reset supplies the
    spawn pose. The term never owns the physical reset operation.
    """

    RESET_STATE_CHILDREN = ("category", "difficulty")

    CHECKPOINT_STATE_FIELDS = {
        "spawn_generations": "_spawn_generations",
        "requested_reallocations": "_requested_reallocations",
        "requested_spawns": "_requested_spawns",
    }

    def __init__(
        self,
        name: str,
        num_envs: int,
        context: RuntimeContext,
        curriculum_manager: CurriculumTermProvider | None = None,
        **params,
    ):

        super().__init__(name=name, num_envs=num_envs, context=context, **params)

        self.config = self.normalize_config(params)
        self.category = TerrainCategoryTerm(
            name=f"{name}_category", config=self.config.category, context=context,
            curriculum_manager=curriculum_manager, num_envs=num_envs,
        )
        self.difficulty = TerrainDifficultyTerm(
            name=f"{name}_difficulty", config=self.config.difficulty, context=context,
            curriculum_manager=curriculum_manager, num_envs=num_envs,
        )
        self.training = True
        self._spawn_generations = [0] * num_envs
        self._requested_reallocations: set[int] = set()
        self._requested_spawns: set[int] = set()
        self._restored_envs: set[int] = set()

        is_fixed_plane = (
            self.config.is_plane
            and self.config.category.is_fixed
            and self.config.difficulty.is_fixed
        )
        self.assignments = (
            None if is_fixed_plane else TerrainAssignments(self.config, context.seed, num_envs)
        )

        self.geometry = self.geometry_from_config(self.config)
        self.simulator: TerrainBackend
        self._static_backend: StaticTerrainBackend | None = None
        self._terrain_layout: tuple[int, tuple[int, int], float] | None = None
        self._patches: list[TerrainAssignment | None] = (
            [None] * num_envs if self.assignments is not None else []
        )
        self._pending: _PendingTerrainReset | None = None


    def _ensure_no_pending_reset(self) -> None:
        if self._pending is not None:
            raise RuntimeError("Finish or abort the pending terrain reset first.")


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
        for params in config.category.values.values():
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
            catalog=_TerrainCatalog(config),
        )


    @property
    def patches(self) -> tuple[TerrainAssignment | None, ...]:
        return tuple(self._patches)


    @classmethod
    def get_curriculum_space_config(
        cls,
        name: str,
        params: Mapping[str, Any],
    ) -> CurriculumSpaceConfigs:

        result: CurriculumSpaceConfigs = {}
        for term_cls, axis in (
            (TerrainCategoryTerm, "category"), (TerrainDifficultyTerm, "difficulty")
        ):
            descriptions = term_cls.get_curriculum_space_config(
                f"{name}_{axis}", params.get(axis)
            )
            for reference, space in descriptions.items():
                previous = result.get(reference, CurriculumSpaceConfig({}))
                result[reference] = CurriculumSpaceConfig(
                    dimensions=previous.dimensions | space.dimensions,
                    persistent_assignments=True,
                )
        return result


    def matches_config(
        self,
        params: Mapping[str, Any],
    ) -> bool:
        return self.config == self.normalize_config(params)


    @classmethod
    def get_curriculum_references(
        cls,
        params: Mapping[str, Any],
    ) -> set[str]:
        return (
            TerrainCategoryTerm.get_curriculum_references(params.get("category"))
            | TerrainDifficultyTerm.get_curriculum_references(params.get("difficulty"))
        )


    def config_update(
        self,
        curriculum_manager: CurriculumTermProvider | None,
        *args, **kwargs
    ) -> None:
        self._ensure_no_pending_reset()
        snapshot = self.category.capture_reset_state()
        snapshot.extend(self.difficulty.capture_reset_state())
        try:
            self.category.config_update(curriculum_manager=curriculum_manager)
            self.difficulty.config_update(curriculum_manager=curriculum_manager)
        except Exception:
            snapshot.restore()
            raise


    def set_training_mode(self, training: bool) -> None:
        self.training = training


    def request_reallocation(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._ensure_no_pending_reset()
        self._requested_reallocations.update(self._selected_ids(env_ids))


    def request_spawn(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._ensure_no_pending_reset()
        self._requested_spawns.update(self._selected_ids(env_ids))


    def _sample_conditions(
        self,
        env_id: int,
    ) -> tuple[str, float]:

        assert self.assignments is not None
        seed = terrain_patch_seed(
            self.assignments.seed, self.assignments.allocation_ids[env_id]
        )
        current = self._patches[env_id]
        category = self.category.sample(
            env_id, seed, None if current is None else current.terrain_type
        )
        difficulty = self.difficulty.sample(
            env_id, seed, None if current is None else current.difficulty
        )
        return category, difficulty


    def _checkpoint_metadata(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Cannot checkpoint an unfinished terrain reset.")
        return {
            "type": type(self).__name__,
            "version": 4,
            "config": self.config.to_mapping(),
            "axes": {
                "category": self.category.checkpoint_state_dict(),
                "difficulty": self.difficulty.checkpoint_state_dict(),
            },
            "num_envs": self.num_envs,
            "seed": self.config.effective_seed(self.context.seed),
            "allocations": None if self.assignments is None else self.assignments.state_dict(),
            "initialized": [patch is not None for patch in self._patches],
            **({"terrain_layout": self._terrain_layout} if self._terrain_layout is not None else {}),
        }


    def validate_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:

        if not isinstance(state, Mapping) or state.get("type") != type(self).__name__:
            raise ValueError(
                "Randomization checkpoint term type does not match configuration."
            )
        if (
            state.get("version") != 4
            or state.get("num_envs") != self.num_envs
            or state.get("seed") != self.config.effective_seed(self.context.seed)
            or state.get("terrain_layout") != self._terrain_layout
            or not isinstance(state.get("config"), Mapping)
            or TerrainConfig.from_mapping(state.get("config")) != self.config
        ):
            raise ValueError(
                "Incompatible terrain checkpoint configuration, "
                "layout, seed, version or environment count."
            )

        spawn_generations = state.get("spawn_generations")
        if (
            not isinstance(spawn_generations, list)
            or len(spawn_generations) != self.num_envs
            or any(type(value) is not int or value < 0 for value in spawn_generations)
        ):
            raise ValueError("Invalid terrain checkpoint spawn_generations.")

        initialized = state.get("initialized")
        count = 0 if self.assignments is None else self.num_envs
        if (
            not isinstance(initialized, list)
            or len(initialized) != count
            or any(type(value) is not bool for value in initialized)
        ):
            raise ValueError("Invalid terrain initialized flags.")

        if self.assignments is None:
            if state.get("allocations") is not None:
                raise ValueError("Plane terrain must not contain geometric allocations.")
        else:
            candidate = deepcopy(self.assignments)
            allocations = state.get("allocations")
            if not isinstance(allocations, Mapping):
                raise ValueError("Terrain checkpoint requires allocation state.")
            candidate.load_state_dict(allocations)

        for name in ("requested_reallocations", "requested_spawns"):
            values = state.get(name)
            if not isinstance(values, list):
                raise ValueError(f"Invalid terrain checkpoint {name}.")
            if (
                any(type(i) is not int or not 0 <= i < self.num_envs for i in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"Invalid terrain checkpoint {name}.")

        axes = state.get("axes")
        if not isinstance(axes, Mapping) or set(axes) != {"category", "difficulty"}:
            raise ValueError("Invalid terrain axis checkpoint state.")

        self.category.validate_checkpoint_state_dict(axes["category"])
        self.difficulty.validate_checkpoint_state_dict(axes["difficulty"])


    def playback_state_dict(
        self,
        env_index: int,
    ) -> dict[str, Any]:

        state = self.checkpoint_state_dict()
        state["num_envs"] = 1
        state["spawn_generations"] = [state["spawn_generations"][env_index]]
        if self.assignments is not None:
            state["initialized"] = [state["initialized"][env_index]]
            state["allocations"] = self.assignments.playback_state_dict(env_index)
        state["axes"] = {
            "category": self.category.playback_state_dict(env_index),
            "difficulty": self.difficulty.playback_state_dict(env_index),
        }
        for name in ("requested_reallocations", "requested_spawns"):
            state[name] = [0] if env_index in state[name] else []
        return state



    def _before_checkpoint_load(self) -> None:
        self._ensure_no_pending_reset()


    def _after_checkpoint_load(
        self,
        state: Mapping[str, Any],
    ) -> None:

        self.category.load_checkpoint_state_dict(state["axes"]["category"])
        self.difficulty.load_checkpoint_state_dict(state["axes"]["difficulty"])
        if self.assignments is not None:
            self.assignments.load_state_dict(state["allocations"])
            self._patches = [
                self._patch_for(self.assignments, i)
                if initialized else None
                for i, initialized in enumerate(state["initialized"])
            ]
        self._restored_envs = {i for i, initialized in enumerate(state["initialized"]) if initialized}
        self._unapplied_envs = set(range(self.num_envs))


    def bind(
        self,
        backend: RandomizationBackend,
    ) -> None:

        self._ensure_no_pending_reset()
        simulator = cast(TerrainBackend, backend)
        if simulator.num_envs != self.num_envs:
            raise ValueError("Terrain term and simulator environment counts differ.")

        if self.geometry is not None:
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
        self._static_backend = (
            backend if self.geometry is not None and isinstance(backend, StaticTerrainBackend) else None
        )
        self._terrain_layout = (
            self._static_backend.get_terrain_layout() if self._static_backend is not None else None
        )
        # A newly configured backend needs the current assignment installed once.
        self._unapplied_envs = set(range(self.num_envs))


    def _patch_for(
        self,
        assignments: TerrainAssignments,
        env_id: int,
    ) -> TerrainAssignment:

        if self._static_backend is None or self._terrain_layout is None:
            return assignments._patch(env_id)
        num_variants = self._terrain_layout[0]
        generation = max(0, assignments.generations[env_id] - 1)
        variant = (assignments.allocation_ids[env_id] + generation) % num_variants
        category = assignments.categories[env_id]
        difficulty = assignments.difficulties[env_id]
        return TerrainAssignment(
            patch_id=variant,
            seed=terrain_patch_seed(assignments.seed, variant),
            terrain_type=category,
            difficulty=difficulty,
            geometry=self._static_backend.get_terrain_patch((category, difficulty, variant)),
            spawn_size=self.config.spawn_size,
        )


    def _selected_ids(
        self,
        env_ids: torch.Tensor | None,
    ) -> list[int]:

        if env_ids is None:
            return list(range(self.num_envs))
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError("Environment IDs must be a one-dimensional long tensor.")
        env_ids_list = env_ids.tolist()
        if (
            len(set(env_ids_list)) != len(env_ids_list)
            or any(env_id < 0 or env_id >= self.num_envs for env_id in env_ids_list)
        ):
            raise ValueError("Environment IDs must be unique and in range.")
        return env_ids_list


    def before_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Compose resolved selections and stage their physical terrain update."""

        self._ensure_no_pending_reset()
        env_ids_list = self._selected_ids(env_ids)
        if self.assignments is None or not env_ids_list:
            return

        candidate = deepcopy(self.assignments)
        spawn_generations = self._spawn_generations.copy()
        changed_env_ids_list = []
        selected_categories = []
        selected_difficulties = []
        for env_id in env_ids_list:
            restored = env_id in self._restored_envs
            initialized = self._patches[env_id] is not None
            if restored or (not self.training and initialized):
                category = candidate.categories[env_id]
                difficulty = candidate.difficulties[env_id]
            else:
                category, difficulty = self._sample_conditions(env_id)
                if self.training and (self.config.resample_spawn or env_id in self._requested_spawns):
                    spawn_generations[env_id] += 1

            selection_changed = (
                not initialized
                or category != candidate.categories[env_id]
                or difficulty != candidate.difficulties[env_id]
            )
            regenerate_requested = (
                self.training and not restored
                and (self.config.resample_patch or env_id in self._requested_reallocations)
            )
            if selection_changed or regenerate_requested:
                changed_env_ids_list.append(env_id)
                selected_categories.append(category)
                selected_difficulties.append(difficulty)

        candidate._reset(changed_env_ids_list, selected_difficulties, selected_categories)
        changed = set(changed_env_ids_list)
        patches = tuple(
            self._patch_for(candidate, env_id)
            if env_id in changed else cast(TerrainAssignment, self._patches[env_id])
            for env_id in env_ids_list
        )
        env_ids_tensor = torch.tensor(env_ids_list, dtype=torch.long, device=self.simulator.context.device)
        snapshot = (
            self.simulator.capture_reset_state(env_ids_tensor)
            if self.geometry is not None else None
        )
        self._pending = _PendingTerrainReset(
            env_ids_tuple=tuple(env_ids_list),
            candidate=candidate,
            patches=patches,
            physical_state=snapshot,
            spawn_generations=spawn_generations,
            restored_envs=frozenset(self._restored_envs & set(env_ids_list)),
        )

        try:
            updates = [(env_id, patch) for env_id, patch in zip(env_ids_list, patches)
                       if env_id in changed or env_id in self._unapplied_envs]
            if updates and self.geometry is not None:
                selected = torch.tensor(
                    [env_id for env_id, _ in updates], dtype=torch.long, device=env_ids_tensor.device,
                )
                if self._static_backend is None:
                    self.simulator.apply_terrain(selected, [patch.geometry for _, patch in updates])
                else:
                    self._static_backend.select_terrain(
                        selected,
                        [
                            (patch.terrain_type, patch.difficulty, patch.patch_id)
                            for _, patch in updates
                        ]
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
        if self.assignments is None or (self._pending is None and not env_ids_list):
            return
        if self._pending is None or self._pending.env_ids_tuple != tuple(env_ids_list):
            raise RuntimeError("Terrain post-reset must match a pending pre-reset.")
        if self.geometry is None:
            return
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
            if self.training and env_id not in pending.restored_envs:
                self.category.commit_sample(env_id)
                self.difficulty.commit_sample(env_id)
                self._requested_reallocations.discard(env_id)
                self._requested_spawns.discard(env_id)
            self._unapplied_envs.discard(env_id)
            self._restored_envs.discard(env_id)
        self._spawn_generations = pending.spawn_generations
        self._pending = None


    def abort_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        if self._pending is None:
            return
        pending = self._pending
        if pending.physical_state is not None:
            self.simulator.restore_reset_state(pending.physical_state)
        self._pending = None


    def set_difficulty(
        self,
        env_ids_tensor: torch.Tensor,
        difficulties: torch.Tensor,
    ) -> None:
        """Request configured difficulty values for the next reset."""

        self._ensure_no_pending_reset()
        if self.assignments is None:
            raise ValueError("Difficulty updates require geometric terrain.")
        self.difficulty.request(self._selected_ids(env_ids_tensor), difficulties)


    def _spawn_positions(
        self,
        env_ids_tensor: torch.Tensor,
        patches: Sequence[TerrainAssignment],
    ) -> torch.Tensor:

        measured = self._spawn_geometry
        xmin, xmax, ymin, ymax = self._spawn_bounds
        generations = (
            self._spawn_generations if self._pending is None
            else self._pending.spawn_generations
        )
        seeds = []
        for env_id, patch in zip(env_ids_tensor.tolist(), patches, strict=True):
            seed = patch.seed
            if self._static_backend is not None:
                assert self.assignments is not None
                seed = terrain_patch_seed(seed, self.assignments.allocation_ids[env_id])
            generation = generations[env_id]
            seeds.append(seed if generation == 0 else terrain_patch_seed(seed, generation))
        positions = torch.tensor([
            [xmin + (xmax - xmin) * _uniform(seed, "spawn:x"),
             ymin + (ymax - ymin) * _uniform(seed, "spawn:y"),
             measured.root_position[2] + self._spawn_lift - measured.origin[2]]
            for seed in seeds
        ], dtype=torch.float64, device=self.simulator.context.device)
        bounds = self.simulator.get_ground_bounds(env_ids_tensor)
        if bounds is None:
            raise RuntimeError(
                "Terrain spawn requires ground bounds for the selected environments."
            )
        lower, upper = bounds
        positions[:, :2] += (lower + upper) / 2
        height = self.simulator.get_ground_heights(
            positions[:, None, :2], env_ids_tensor
        )
        positions[:, 2] += height[:, 0]
        return positions
