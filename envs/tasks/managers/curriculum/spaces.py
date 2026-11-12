"""Combine manager-owned descriptions and construct curriculum candidate tables."""
import math
import torch
from copy import deepcopy
from collections.abc import Collection, Sequence

from .types import (
    ContinuousDimensionConfig, DiscreteDimensionConfig,
    CurriculumSpace, CurriculumSpaceConfig, CurriculumSpaceConfigs,
    CurriculumDimensionConfig,
)
from utils import RuntimeContext


_MAX_CELLS = 100_000


def build_curriculum_spaces(
    contributions: Sequence[CurriculumSpaceConfigs],
    curriculum_names: Collection[str],
    context: RuntimeContext,
) -> dict[str, CurriculumSpace]:

    spaces: CurriculumSpaceConfigs = {}
    for contribution in contributions:
        for curriculum_name, space in contribution.items():
            if curriculum_name not in curriculum_names:
                raise ValueError(
                    f"Space references unknown curriculum '{curriculum_name}'."
                )
            if not space.dimensions:
                raise ValueError(
                    f"Curriculum space '{curriculum_name}' needs dimensions."
                )

            previous = spaces.get(curriculum_name, CurriculumSpaceConfig({}))
            duplicates = previous.dimensions.keys() & space.dimensions.keys()
            if duplicates:
                raise ValueError(
                    "Duplicate dimensions "
                    f"in curriculum '{curriculum_name}': {sorted(duplicates)}."
                )
            merged_source_config = deepcopy(previous.source_config)
            for source_name, source_config in space.source_config.items():
                if (
                    source_name in merged_source_config
                    and merged_source_config[source_name] != source_config
                ):
                    raise ValueError(
                        f"Conflicting curriculum source configuration: {source_name}."
                    )
                merged_source_config[source_name] = deepcopy(source_config)
            spaces[curriculum_name] = CurriculumSpaceConfig(
                dimensions=previous.dimensions | space.dimensions,
                filters=previous.filters + space.filters,
                source_config=merged_source_config,
            )

    result = {}
    for curriculum_name, space in spaces.items():
        result[curriculum_name] = _build_space(context, curriculum_name, space)
    return result


def _build_space(
    context: RuntimeContext,
    name: str,
    space: CurriculumSpaceConfig,
) -> CurriculumSpace:

    axes, cell_interval, dimension_names, term_slices = _build_dimensions(
        context, name, space.dimensions,
    )

    # Construct joint candidates once, after all manager contributions are merged.
    all_cell_starts = (
        axes[0].unsqueeze(-1)
        if len(axes) == 1
        else torch.cartesian_prod(*axes)
    )
    all_starts = {
        term_name: all_cell_starts[:, term_slice]
        for term_name, term_slice in term_slices.items()
    }
    intervals = {
        term_name: cell_interval[term_slice]
        for term_name, term_slice in term_slices.items()
    }
    filtered_cell_starts = _filter_cells(name, space, all_starts, intervals)

    return CurriculumSpace(
        dimension_names=dimension_names,
        term_slices=term_slices,
        cell_starts=filtered_cell_starts,
        cell_interval=cell_interval,
        source_config=space.source_config,
    )


def _build_dimensions(
    context: RuntimeContext,
    name: str,
    dimensions: dict[str, CurriculumDimensionConfig],
) -> tuple[list[torch.Tensor], torch.Tensor, tuple[str, ...], dict[str, slice]]:
    """Validate dimensions and build their axes, intervals and column mapping."""

    axes: list[torch.Tensor] = []
    cell_intervals: list[torch.Tensor] = []
    dimension_names: list[str] = []
    term_slices: dict[str, slice] = {}

    raw_num_cells = 1
    for term_name, config in dimensions.items():
        if isinstance(config, DiscreteDimensionConfig):
            term_axes, term_intervals, raw_num_cells = _build_discrete_dimension(
                context, name, term_name, config, raw_num_cells,
            )
        elif isinstance(config, ContinuousDimensionConfig):
            term_axes, term_intervals, raw_num_cells = _build_continuous_dimension(
                context, name, term_name, config, raw_num_cells,
            )
        else:
            raise TypeError(f"Unsupported curriculum dimension: {term_name}.")

        start = len(axes)
        dimension_size = len(term_axes)
        axes.extend(term_axes)
        cell_intervals.append(term_intervals)
        term_slices[term_name] = slice(start, start + dimension_size)
        dimension_names.extend(
            [term_name]
            if dimension_size == 1
            else [f"{term_name}[{index}]" for index in range(dimension_size)]
        )

    return axes, torch.cat(cell_intervals), tuple(dimension_names), term_slices


def _build_discrete_dimension(
    context: RuntimeContext,
    name: str,
    term_name: str,
    config: DiscreteDimensionConfig,
    raw_num_cells: int,
) -> tuple[list[torch.Tensor], torch.Tensor, int]:

    values = config.values
    if (
        not values
        or len(set(values)) != len(values)
        or any(not math.isfinite(value) for value in values)
    ):
        raise ValueError(
            f"Discrete dimension '{term_name}' needs unique finite values."
        )

    raw_num_cells *= len(values)
    if raw_num_cells > _MAX_CELLS:
        raise ValueError(
            f"Curriculum space '{name}' exceeds the internal cell limit of {_MAX_CELLS}."
        )

    axis = torch.tensor(values, dtype=context.dtype, device=context.device)
    intervals = torch.zeros(1, dtype=context.dtype, device=context.device)
    return [axis], intervals, raw_num_cells


def _build_continuous_dimension(
    context: RuntimeContext,
    name: str,
    term_name: str,
    config: ContinuousDimensionConfig,
    raw_num_cells: int,
) -> tuple[list[torch.Tensor], torch.Tensor, int]:

    num_cell = config.num_cell
    if type(num_cell) is not int or num_cell < 1:
        raise ValueError(f"'num_cell' of dimension '{term_name}' must be positive.")

    dimension_size = config.dim
    if (
        not isinstance(dimension_size, int)
        or isinstance(dimension_size, bool)
        or dimension_size <= 0
    ):
        raise ValueError(
            f"'dimension_size' of dimension '{term_name}' must be a positive integer."
        )

    raw_num_cells *= num_cell ** dimension_size
    if raw_num_cells > _MAX_CELLS:
        raise ValueError(
            f"Curriculum space '{name}' has at least "
            f"{raw_num_cells} raw cells, exceeding "
            f"the internal cell limit of {_MAX_CELLS}. Reduce 'num_cell' or "
            "'dimension_size', or split the dimensions into "
            "different curriculum instances."
        )

    min_value = float(config.min_value)
    max_value = float(config.max_value)
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise ValueError(f"Dimension '{term_name}' bounds must be finite.")
    if min_value > max_value:
        raise ValueError(
            f"'min_value' of dimension '{term_name}' cannot exceed 'max_value'."
        )

    interval = (max_value - min_value) / num_cell
    intervals = torch.full(
        (dimension_size,), interval, dtype=context.dtype, device=context.device,
    )
    axis = (
        torch.arange(num_cell, dtype=context.dtype, device=context.device) * interval
        + min_value
    )
    return [axis] * dimension_size, intervals, raw_num_cells


def _filter_cells(
    name: str,
    space: CurriculumSpaceConfig,
    all_starts: dict[str, torch.Tensor],
    intervals: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply constraints and validate the remaining joint candidates."""

    # Keep only cells whose complete intervals satisfy every constraint.
    filtered_starts = all_starts
    for filter_cells in space.filters:

        filtered_starts = filter_cells(filtered_starts, intervals)
        if set(filtered_starts) != set(space.dimensions):
            raise ValueError("A curriculum filter must preserve every dimension.")

        row_counts = set()
        for field_name, starts in filtered_starts.items():
            if (
                not isinstance(starts, torch.Tensor)
                or starts.ndim != 2
                or starts.shape[1] != all_starts[field_name].shape[1]
            ):
                raise ValueError("A curriculum filter must preserve dimension widths.")
            row_counts.add(starts.shape[0])
        if len(row_counts) != 1:
            raise ValueError("A curriculum filter must keep all dimensions aligned.")

    filtered_cell_starts = torch.cat(
        [filtered_starts[name] for name in space.dimensions],
        dim=-1,
    )

    num_cells = filtered_cell_starts.shape[0]
    if num_cells == 0:
        raise ValueError(
            f"Curriculum space '{name}' has no cells "
            "remained after applying constraints."
        )
    if num_cells > _MAX_CELLS:
        raise ValueError(
            f"Curriculum space '{name}' has {num_cells} cells, "
            f"exceeding the internal cell limit of {_MAX_CELLS}."
        )

    return filtered_cell_starts
