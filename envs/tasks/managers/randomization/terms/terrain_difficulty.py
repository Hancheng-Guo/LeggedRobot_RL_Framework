"""Terrain difficulty candidates and fixed, uniform or curriculum selection."""
from __future__ import annotations

import torch
import math
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from envs.tasks.managers.curriculum import (
    CurriculumSpaceConfig, CurriculumSpaceConfigs, CurriculumTermProvider,
    DiscreteDimensionConfig, resolve_curriculum_samplers,
)
from envs.tasks.utils import StateOwner
from utils import RuntimeContext


def difficulty_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Terrain difficulty must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Terrain difficulty must be in [0, 1].")
    return result


@dataclass(frozen=True)
class TerrainDifficultyConfig:

    values: tuple[float, ...] = (1.,)
    type: str = "Fixed"
    curriculum: str | None = None


    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> TerrainDifficultyConfig:

        if raw is None:
            return cls()

        if not isinstance(raw, Mapping) or set(raw) - {"type", "params"}:
            raise ValueError("terrain.difficulty requires type and optional params.")
        
        params = raw.get("params", {})
        if not isinstance(params, Mapping):
            raise TypeError("Terrain difficulty params must be a mapping.")

        raw_values = params.get("values", (1.,))
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise TypeError("terrain.difficulty.params.values must be a sequence.")
        values = tuple(difficulty_value(item) for item in raw_values)
        if not values or len(set(values)) != len(values):
            raise ValueError("Terrain difficulty values must be non-empty and unique.")

        allowed = {"values"}
        strategy = raw.get("type")
        if strategy == "Fixed":
            if len(values) != 1:
                raise ValueError(
                    "Fixed terrain difficulty requires exactly one candidate in values."
                )
            config = cls(type="Fixed", values=values)
        elif strategy == "Uniform":
            config = cls(type="Uniform", values=values)
        elif strategy == "Curriculum":
            allowed.add("curriculum")
            curriculum = params.get("curriculum")
            if not isinstance(curriculum, str) or not curriculum:
                raise ValueError(
                    "Terrain curriculum sampling requires an explicit curriculum instance."
                )
            config = cls(type="Curriculum", values=values, curriculum=curriculum)
        else:
            raise ValueError("Terrain axis type must be Fixed, Uniform or Curriculum.")

        if set(params) - allowed:
            raise ValueError("Unsupported parameters for terrain difficulty sampling.")
        return config


    @property
    def is_fixed(self) -> bool:
        return self.type == "Fixed"


    @property
    def initial_value(self) -> float:
        return self.values[0]


    def resolve_index(self, index: int) -> float:
        if type(index) is not int or not 0 <= index < len(self.values):
            raise ValueError("Invalid terrain difficulty index.")
        return self.values[index]


    def validate_selection(self, value: Any) -> float:
        value = difficulty_value(value)
        if value not in self.values:
            raise ValueError("Terrain difficulty must match one configured value.")
        return value


    def to_mapping(self) -> dict[str, Any]:
        params: dict[str, Any] = {"values": list(self.values)}
        if self.curriculum is not None:
            params["curriculum"] = self.curriculum
        return {"type": self.type, "params": params}


class TerrainDifficultyTerm(StateOwner):
    """Own difficulty selection; terrain owns patch construction and persistence."""

    STATE_FIELDS = {"sample_counts": "_sample_counts", "requests": "_requests"}
    RESET_STATE_FIELDS = ("_curriculum_samplers",)


    def __init__(
        self,
        name: str,
        config: TerrainDifficultyConfig,
        context: RuntimeContext,
        curriculum_manager: CurriculumTermProvider | None = None,
        num_envs: int = 1,
    ) -> None:

        super().__init__(name=name)

        self.config = config
        self.context = context
        self.num_envs = num_envs
        self._sample_counts = [0] * num_envs
        self._requests: dict[int, float] = {}
        self.config_update(curriculum_manager=curriculum_manager)


    @classmethod
    def get_curriculum_references(
        cls,
        params: Mapping[str, Any] | None,
    ) -> set[str]:
        config = TerrainDifficultyConfig.from_mapping(params)
        return set() if config.curriculum is None else {config.curriculum}


    @classmethod
    def get_curriculum_space_config(
        cls,
        name: str,
        params: Mapping[str, Any] | None,
    ) -> CurriculumSpaceConfigs:
        config = TerrainDifficultyConfig.from_mapping(params)
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

        references = set() if self.config.curriculum is None else {self.config.curriculum}
        self._curriculum_samplers = resolve_curriculum_samplers(
            references, curriculum_manager, f"Terrain axis {self.name!r}",
        )


    def _curriculum_index(
        self,
        env_id: int,
    ) -> int:

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


    def sample(
        self,
        env_id: int,
        seed: int,
        current: float | None = None,
    ) -> float:

        if env_id in self._requests:
            return self._requests[env_id]

        if self.config.type == "Fixed":
            return self.config.initial_value

        if self.config.type == "Curriculum":
            return self.config.resolve_index(self._curriculum_index(env_id))

        label = f"sampling:{self._sample_counts[env_id]}:difficulty"
        digest = hashlib.blake2b(
            f"terrain-v1:{seed}:{label}".encode("ascii"), digest_size=8
        ).digest()
        draw = (int.from_bytes(digest, "little") >> 11) / (1 << 53)
        index = int(draw * len(self.config.values))
        return self.config.resolve_index(index)


    def commit_sample(self, env_id: int) -> None:
        """Advance sampling only after the complete terrain reset succeeds."""

        if self.config.type == "Uniform":
            self._sample_counts[env_id] += 1
        self._requests.pop(env_id, None)


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
            or TerrainDifficultyConfig.from_mapping(state.get("config")) != self.config
        ):
            raise ValueError("Incompatible terrain difficulty state.")

        counts = state.get("sample_counts")
        if (
            not isinstance(counts, list)
            or len(counts) != self.num_envs
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise ValueError("Invalid terrain difficulty sampling counts.")

        requests = state.get("requests")
        if (
            not isinstance(requests, Mapping)
            or (self.config.type == "Curriculum" and requests)
        ):
            raise ValueError("Invalid pending terrain difficulty requests.")

        for env_id, value in requests.items():
            if type(env_id) is not int or not 0 <= env_id < self.num_envs:
                raise ValueError("Invalid pending terrain difficulty requests.")
            self.config.validate_selection(value)


    def playback_state_dict(self, env_index: int) -> dict[str, Any]:

        if not 0 <= env_index < self.num_envs:
            raise ValueError("Playback environment index is out of range.")
        state = self.checkpoint_state_dict()
        state["num_envs"] = 1
        state["sample_counts"] = [state["sample_counts"][env_index]]
        state["requests"] = (
            {0: state["requests"][env_index]}
            if env_index in state["requests"]
            else {}
        )
        return state


    def request(
        self,
        env_ids_list: Sequence[int],
        difficulties: torch.Tensor,
    ) -> None:
        """Validate and retain actual difficulty values until a successful reset."""

        if self.config.type == "Curriculum":
            raise ValueError("Curriculum sampling owns terrain difficulty selection.")
        if (
            difficulties.ndim != 1
            or len(env_ids_list) != difficulties.numel()
            or difficulties.dtype == torch.bool
            or difficulties.is_complex()
            or not torch.isfinite(difficulties).all()
        ):
            raise ValueError("Invalid terrain difficulty values.")

        values = (
            difficulties if difficulties.is_floating_point()
            else difficulties.to(torch.float64)
        )
        candidates = values.new_tensor(self.config.values)
        matches = values[:, None] == candidates[None, :]
        if not (matches.sum(dim=1) == 1).all():
            raise ValueError("Terrain difficulty must match one configured value.")
        indices = (
            matches.to(torch.long).argmax(dim=1).tolist() if env_ids_list
            else []
        )
        requests = {
            env_id: self.config.resolve_index(index)
            for env_id, index in zip(env_ids_list, indices)
        }
        self._requests.update(requests)
