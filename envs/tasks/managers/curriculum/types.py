import torch
from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class ContinuousDimensionConfig:
    min_value: float
    max_value: float
    num_cell: int
    dim: int = 1


@dataclass(frozen=True)
class DiscreteDimensionConfig:
    # Consumers own the mapping from numeric choices to domain-specific labels.
    values: tuple[float, ...]


CurriculumDimensionConfig = ContinuousDimensionConfig | DiscreteDimensionConfig
CurriculumCellFilter = Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]], dict[str, torch.Tensor]
]


@dataclass(frozen=True)
class CurriculumSpaceConfig:
    """One manager's contribution, before joint cells are constructed."""
    dimensions: dict[str, CurriculumDimensionConfig]
    filters: tuple[CurriculumCellFilter, ...] = ()
    source_config: dict[str, Any] = field(default_factory=dict)


CurriculumSpaceConfigs = dict[str, CurriculumSpaceConfig]


@dataclass(frozen=True)
class CurriculumSpace:
    """Joint candidate intervals built from manager-owned descriptions."""
    dimension_names: tuple[str, ...]
    term_slices: dict[str, slice]
    cell_starts: torch.Tensor
    cell_interval: torch.Tensor
    # Opaque source configuration participates in checkpoint compatibility checks.
    source_config: dict[str, Any] = field(default_factory=dict)


class CurriculumSpaceProvider(Protocol):

    @classmethod
    def get_curriculum_space_config(
        cls, config: Mapping[str, Any] | None,
    ) -> CurriculumSpaceConfigs:
        ...


@runtime_checkable
class CurriculumTermProvider(Protocol):

    def get_term(self, name: str) -> Any:
        ...


@runtime_checkable
class CommandCurriculumSampler(Protocol):

    def get_command(
        self,
        dimension: str,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
