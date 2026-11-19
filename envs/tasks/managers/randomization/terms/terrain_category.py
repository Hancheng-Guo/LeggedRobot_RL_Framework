"""Terrain category candidates and fixed, weighted or curriculum selection."""
from __future__ import annotations

import torch
import math
import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias

from envs.tasks.managers.curriculum import (
    CurriculumSpaceConfig, CurriculumSpaceConfigs, CurriculumTermProvider,
    DiscreteDimensionConfig, resolve_curriculum_samplers,
)
from envs.tasks.utils import StateOwner
from utils import RuntimeContext


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


def finite_number(
    value: Any,
    name: str,
) -> float:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"terrain.{name} must contain finite numbers.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"terrain.{name} must contain finite numbers.")
    return result


def _category_values(
    raw: Any,
    strategy: str,
) -> dict[str, TerrainTypeConfig]:

    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("terrain.category.params.values must be a non-empty mapping.")
    if strategy == "Fixed" and len(raw) != 1:
        raise ValueError("Fixed terrain category requires exactly one candidate in values.")

    result: dict[str, TerrainTypeConfig] = {}
    for name, params in raw.items():
        if not isinstance(params, Mapping):
            raise TypeError(f"terrain.category.params.values.{name} must be a mapping.")
        if strategy != "Weighted" and "weight" in params:
            raise ValueError(f"Terrain category {strategy} does not accept weight for {name!r}.")

        allowed = {"weight"} if strategy == "Weighted" else set()
        if name == "slope":
            allowed.add("max_slope")
        elif name == "rough":
            allowed.add("max_height")
        elif name != "plane":
            raise ValueError(f"Unsupported terrain category: {name}.")
        if set(params) - allowed:
            raise ValueError(
                f"Unknown terrain category {name} field(s): {sorted(set(params) - allowed)}."
            )

        weight = finite_number(params.get("weight", 1.), f"category.{name}.weight")
        if weight <= 0:
            raise ValueError("Terrain category weight must be positive.")

        if name == "slope":
            slope = finite_number(params.get("max_slope", 0.), "category.slope.max_slope")
            if slope <= 0:
                raise ValueError("terrain.category.slope.max_slope must be positive.")
            result[name] = SlopeTerrainConfig(max_slope=slope, weight=weight)
        elif name == "rough":
            height = finite_number(params.get("max_height", 0.), "category.rough.max_height")
            if height <= 0:
                raise ValueError("terrain.category.rough.max_height must be positive.")
            result[name] = RoughTerrainConfig(max_height=height, weight=weight)
        else:
            result[name] = PlaneTerrainConfig(weight=weight)

    return result


@dataclass(frozen=True)
class TerrainCategoryConfig:

    values: dict[str, TerrainTypeConfig] = field(
        default_factory=lambda: {"plane": PlaneTerrainConfig()}
    )
    type: str = "Fixed"
    curriculum: str | None = None


    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> TerrainCategoryConfig:

        if raw is None:
            return cls()
        if not isinstance(raw, Mapping) or set(raw) - {"type", "params"}:
            raise ValueError("terrain.category requires type and optional params.")
        strategy = raw.get("type")
        if strategy not in {"Fixed", "Weighted", "Curriculum"}:
            raise ValueError("Terrain axis type must be Fixed, Weighted or Curriculum.")
        params = raw.get("params", {})
        if not isinstance(params, Mapping):
            raise TypeError("Terrain category params must be a mapping.")

        allowed = {"values"}
        curriculum = None
        if strategy == "Curriculum":
            allowed.add("curriculum")
            curriculum = params.get("curriculum")
            if not isinstance(curriculum, str) or not curriculum:
                raise ValueError(
                    "Terrain curriculum sampling requires an explicit curriculum instance."
                )
        if set(params) - allowed:
            raise ValueError("Unsupported parameters for terrain category sampling.")

        values = _category_values(params.get("values", {"plane": {}}), strategy)
        return cls(type=strategy, values=values, curriculum=curriculum)


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TerrainCategoryConfig):
            return NotImplemented
        return tuple(self.values) == tuple(other.values) and asdict(self) == asdict(other)


    @property
    def is_plane(self) -> bool:
        return tuple(self.values) == ("plane",)


    @property
    def is_fixed(self) -> bool:
        return self.type == "Fixed"


    @property
    def initial_value(self) -> str:
        return next(iter(self.values))


    def resolve_index(self, index: int) -> str:
        if type(index) is not int or not 0 <= index < len(self.values):
            raise ValueError("Invalid terrain category index.")
        return tuple(self.values)[index]


    def parameters_for(self, category: str) -> TerrainTypeConfig:
        return self.values[self.validate_selection(category)]


    def validate_selection(self, value: Any) -> str:
        if not isinstance(value, str) or value not in self.values:
            raise ValueError("Invalid terrain category selection.")
        return value


    def to_mapping(self) -> dict[str, Any]:
        values = {name: asdict(value) for name, value in self.values.items()}
        if self.type != "Weighted":
            for candidate in values.values():
                candidate.pop("weight", None)
        params: dict[str, Any] = {"values": values}
        if self.curriculum is not None:
            params["curriculum"] = self.curriculum
        return {"type": self.type, "params": params}


class TerrainCategoryTerm(StateOwner):
    """Own category selection; terrain owns patch construction and persistence."""

    STATE_FIELDS = {"sample_counts": "_sample_counts"}
    RESET_STATE_FIELDS = ("_curriculum_samplers",)

    def __init__(
        self,
        name: str,
        config: TerrainCategoryConfig,
        context: RuntimeContext,
        curriculum_manager: CurriculumTermProvider | None = None,
        num_envs: int = 1,
    ) -> None:

        super().__init__(name=name)

        self.config = config
        self.context = context
        self.num_envs = num_envs
        self._sample_counts = [0] * num_envs
        self.config_update(curriculum_manager=curriculum_manager)


    @classmethod
    def get_curriculum_references(
        cls,
        params: Mapping[str, Any] | None,
    ) -> set[str]:
        config = TerrainCategoryConfig.from_mapping(params)
        return set() if config.curriculum is None else {config.curriculum}


    @classmethod
    def get_curriculum_space_config(
        cls,
        name: str,
        params: Mapping[str, Any] | None,
    ) -> CurriculumSpaceConfigs:
        config = TerrainCategoryConfig.from_mapping(params)
        if config.curriculum is None:
            return {}
        return {config.curriculum: CurriculumSpaceConfig(
            dimensions={
                name: DiscreteDimensionConfig(
                    tuple(float(i) for i in range(len(config.values)))
                )
            },
            persistent_assignments=True,
        )}


    def config_update(
        self,
        curriculum_manager: CurriculumTermProvider | None,
        *args, **kwargs,
    ) -> None:
        references = (
            set() if self.config.curriculum is None
            else {self.config.curriculum}
        )
        self._curriculum_samplers = resolve_curriculum_samplers(
            references, curriculum_manager, f"Terrain axis {self.name!r}",
        )


    def _curriculum_index(self, env_id: int) -> int:
        assert self.config.curriculum is not None
        env_ids_tensor = torch.tensor(
            [env_id], dtype=torch.long, device=self.context.device
        )
        lower, upper = self._curriculum_samplers[self.config.curriculum].get_bounds(
            self.name, env_ids_tensor,
        )
        if lower.numel() != 1 or not torch.equal(lower, upper):
            raise ValueError(
                "Terrain curriculum dimensions must contain discrete indices."
            )
        value = lower.item()
        if (
            not math.isfinite(value)
            or not float(value).is_integer()
            or not 0 <= value < len(self.config.values)
        ):
            raise ValueError(
                "Terrain curriculum dimensions must contain valid discrete indices."
            )
        return int(value)


    def _weighted_index(self, draw: float) -> int:
        maximum = max(params.weight for params in self.config.values.values())
        weights = [params.weight / maximum for params in self.config.values.values()]
        remaining = draw * sum(weights)
        for index, weight in enumerate(weights):
            remaining -= weight
            if remaining < 0:
                return index
        return len(weights) - 1


    def sample(
        self,
        env_id: int,
        seed: int,
        current: str | None = None,
    ) -> str:

        if self.config.type == "Fixed":
            return self.config.initial_value
        if self.config.type == "Curriculum":
            return self.config.resolve_index(self._curriculum_index(env_id))

        label = f"sampling:{self._sample_counts[env_id]}:category"
        digest = hashlib.blake2b(
            f"terrain-v1:{seed}:{label}".encode("ascii"), digest_size=8
        ).digest()
        draw = (int.from_bytes(digest, "little") >> 11) / (1 << 53)
        return self.config.resolve_index(self._weighted_index(draw))


    def commit_sample(self, env_id: int) -> None:
        """Advance sampling only after the complete terrain reset succeeds."""
        if self.config.type == "Weighted":
            self._sample_counts[env_id] += 1


    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "config": self.config.to_mapping(),
        }


    def validate_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("num_envs") != self.num_envs
            or not isinstance(state.get("config"), Mapping)
            or TerrainCategoryConfig.from_mapping(state.get("config")) != self.config
        ):
            raise ValueError("Incompatible terrain category state.")
        counts = state.get("sample_counts")
        if (
            not isinstance(counts, list)
            or len(counts) != self.num_envs
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise ValueError("Invalid terrain category sampling counts.")


    def playback_state_dict(self, env_index: int) -> dict[str, Any]:
        if not 0 <= env_index < self.num_envs:
            raise ValueError("Playback environment index is out of range.")
        state = self.checkpoint_state_dict()
        state["num_envs"] = 1
        state["sample_counts"] = [state["sample_counts"][env_index]]
        return state
