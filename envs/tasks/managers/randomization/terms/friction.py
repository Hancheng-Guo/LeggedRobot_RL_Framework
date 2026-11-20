"""Sample and persist one sliding friction factor per environment."""
import torch
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import blake2b
from math import isfinite
from typing import Any

from .base import BaseRandomizationTerm, RandomizationBackend
from .registry import register_randomization
from envs.simulators.friction import FrictionBackend
from envs.tasks.managers.curriculum import (
    CurriculumSpaceConfig, CurriculumSpaceConfigs, CurriculumTermProvider,
    DiscreteDimensionConfig, resolve_curriculum_samplers,
)
from utils import RuntimeContext


@dataclass(frozen=True)
class FrictionConfig:

    type: str
    factors: tuple[float, ...]
    curriculum: str | None
    seed: int | None


    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "FrictionConfig":

        if set(config) - {"type", "params", "seed"}:
            raise ValueError("Unsupported friction configuration fields.")
        strategy = config.get("type")
        if strategy not in ("Fixed", "Uniform", "Curriculum"):
            raise ValueError("Friction type must be Fixed, Uniform or Curriculum.")
        seed = config.get("seed")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("Friction seed must be a non-negative integer or null.")
        params = config.get("params")
        allowed = {"factors", "curriculum"} if strategy == "Curriculum" else {"factors"}
        if not isinstance(params, Mapping) or set(params) - allowed:
            raise ValueError("Invalid friction sampling parameters.")
        raw_factors = params.get("factors")
        if not isinstance(raw_factors, (list, tuple)) or not raw_factors:
            raise ValueError("Friction requires a non-empty list of factors.")
        if any(type(value) not in (int, float) or not isfinite(value) or value < 0
               for value in raw_factors):
            raise ValueError("Friction factors must be finite non-negative numbers.")
        factors = tuple(float(value) for value in raw_factors)
        if len(set(factors)) != len(factors) or (strategy == "Fixed" and len(factors) != 1):
            raise ValueError(
                "Friction candidates must be unique; Fixed requires one candidate."
            )
        curriculum = params.get("curriculum")
        if strategy == "Curriculum" and (not isinstance(curriculum, str) or not curriculum):
            raise ValueError("Friction Curriculum requires an explicit curriculum instance.")
        return cls(type=strategy, factors=factors, curriculum=curriculum, seed=seed)


@dataclass
class _PendingFrictionReset:

    env_ids: list[int]
    factors: list[float]
    advances: list[bool]
    snapshot: object


@register_randomization
class FrictionRandomizationTerm(BaseRandomizationTerm):

    CHECKPOINT_STATE_FIELDS = ("factors", "sample_counts", "sampling_env_ids")


    def __init__(
        self,
        name: str,
        num_envs: int,
        context: RuntimeContext,
        curriculum_manager: CurriculumTermProvider | None = None,
        **params,
    ) -> None:

        super().__init__(name, num_envs, context)

        self.config = FrictionConfig.from_mapping(params)
        self.seed = context.seed if self.config.seed is None else self.config.seed
        self.factors = [-1.0] * num_envs
        self.sample_counts = [0] * num_envs
        self.sampling_env_ids = list(range(num_envs))
        self._restored_envs: set[int] = set()
        self._pending: _PendingFrictionReset | None = None
        self._backend: FrictionBackend | None = None
        self._training = True
        self.config_update(curriculum_manager)


    @classmethod
    def get_curriculum_space_config(
        cls,
        name: str,
        params: Mapping[str, Any],
    ) -> CurriculumSpaceConfigs:

        config = FrictionConfig.from_mapping(params)
        if config.curriculum is None:
            return {}
        return {
            config.curriculum: CurriculumSpaceConfig(
                dimensions={
                    name: DiscreteDimensionConfig(
                        tuple(float(i) for i in range(len(config.factors)))
                )},
                persistent_assignments=True,
        )}


    @classmethod
    def get_curriculum_references(
        cls,
        params: Mapping[str, Any],
    ) -> set[str]:
        curriculum = FrictionConfig.from_mapping(params).curriculum
        return set() if curriculum is None else {curriculum}


    def config_update(
        self,
        curriculum_manager: CurriculumTermProvider | None = None,
        **kwargs,
    ) -> None:

        self._ensure_no_pending_reset()
        curriculum = self.config.curriculum
        references = set() if curriculum is None else {curriculum}
        samplers = resolve_curriculum_samplers(
            references, curriculum_manager, f"Friction term {self.name!r}"
        )
        self._sampler = None if curriculum is None else samplers[curriculum]


    def matches_config(
        self,
        params: Mapping[str, Any],
    ) -> bool:
        return self.config == FrictionConfig.from_mapping(params)


    def bind(
        self,
        backend: RandomizationBackend,
    ) -> None:
        self._ensure_no_pending_reset()
        if not isinstance(backend, FrictionBackend):
            raise NotImplementedError(
                "This simulator does not support friction randomization."
            )
        if backend.num_envs != self.num_envs:
            raise ValueError("Friction term and simulator environment counts differ.")
        backend.prepare_friction()
        self._backend = backend


    def set_training_mode(
        self,
        training: bool,
    ) -> None:
        self._ensure_no_pending_reset()
        self._training = training


    def _ensure_no_pending_reset(self) -> None:
        if self._pending is not None:
            raise RuntimeError("Finish or abort the pending friction reset first.")


    def _sample(
        self,
        env_ids: list[int],
    ) -> list[float]:

        if not env_ids:
            return []
        factors = self.config.factors
        if self._sampler is not None:
            indices = torch.tensor(env_ids, dtype=torch.long, device=self.context.device)
            lower, upper = self._sampler.get_bounds(self.name, indices)
            if lower.shape != (len(env_ids), 1) or not torch.equal(lower, upper):
                raise ValueError(
                    "Friction curriculum requires one discrete index per environment."
                )
            choices = lower[:, 0].tolist()
            if any(
                not isfinite(i) or i != int(i) or not 0 <= i < len(factors)
                for i in choices
            ):
                raise ValueError("Invalid friction curriculum candidate index.")
            return [factors[int(i)] for i in choices]

        if self.config.type == "Fixed":
            return [factors[0]] * len(env_ids)

        result = []
        for env_id in env_ids:
            key = (
                f"friction-v1:{self.seed}:"
                f"{self.sampling_env_ids[env_id]}:{self.sample_counts[env_id]}"
            )
            digest = blake2b(key.encode("ascii"), digest_size=8).digest()
            draw = int.from_bytes(digest, "little")
            candidate_index = (draw * len(factors)) >> 64
            result.append(factors[candidate_index])
        return result


    def before_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        self._ensure_no_pending_reset()
        if self._backend is None:
            raise RuntimeError("Bind friction to a simulator before reset.")
        if env_ids is not None and (env_ids.ndim != 1 or env_ids.dtype != torch.long):
            raise ValueError("Environment IDs must be a one-dimensional long tensor.")
        selected = list(range(self.num_envs)) if env_ids is None else env_ids.tolist()
        if (
            len(set(selected)) != len(selected)
            or any(not 0 <= i < self.num_envs for i in selected)
        ):
            raise ValueError("Environment IDs must be unique and in range.")
        if not selected:
            return

        sample_ids = [
            i for i in selected if self.factors[i] < 0
            or (self._training and i not in self._restored_envs)
        ]
        sampled = dict(zip(sample_ids, self._sample(sample_ids), strict=True))
        factors = [sampled.get(i, self.factors[i]) for i in selected]
        advances = [
            self._training and i in sampled and self.config.type == "Uniform"
            for i in selected
        ]
        snapshot = self._backend.capture_friction(selected)
        self._pending = _PendingFrictionReset(selected, factors, advances, snapshot)
        try:
            self._backend.apply_friction(selected, factors)
        except Exception as error:
            try:
                self.abort_simulator_reset()
            except Exception as restore_error:
                raise ExceptionGroup(
                    "Friction update and rollback failed", [error, restore_error]
                ) from error
            raise


    def commit_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:

        pending = self._pending
        if pending is None:
            return

        for i, factor, advance in zip(
            pending.env_ids, pending.factors, pending.advances
        ):
            self.factors[i] = factor
            self.sample_counts[i] += int(advance)
            self._restored_envs.discard(i)
        self._pending = None


    def abort_simulator_reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if self._pending is not None:
            assert self._backend is not None
            self._backend.restore_friction(self._pending.snapshot)
            self._pending = None


    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {"version": 2, "config": asdict(self.config), "seed": self.seed}


    def checkpoint_state_dict(self) -> dict[str, Any]:
        self._ensure_no_pending_reset()
        return super().checkpoint_state_dict()


    def _before_checkpoint_load(self) -> None:
        self._ensure_no_pending_reset()


    def validate_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        super().validate_checkpoint_state_dict(state)
        if (
            any(
                value != -1.0 and value not in self.config.factors
                for value in state["factors"]
            )
            or any(count < 0 for count in state["sample_counts"])
            or any(i < 0 for i in state["sampling_env_ids"])
            or len(set(state["sampling_env_ids"])) != self.num_envs
        ):
            raise ValueError("Invalid friction assignment state.")


    def _after_checkpoint_load(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self._restored_envs = {
            i for i, factor in enumerate(self.factors) if factor >= 0
        }


    def playback_state_dict(
        self,
        env_index: int,
    ) -> dict[str, Any]:
        if type(env_index) is not int or not 0 <= env_index < self.num_envs:
            raise ValueError("Playback environment index is out of range.")
        state = self.checkpoint_state_dict()
        for field in self.CHECKPOINT_STATE_FIELDS:
            state[field] = [state[field][env_index]]
        return state
