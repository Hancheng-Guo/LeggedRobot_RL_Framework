import re
import torch
from dataclasses import dataclass
from typing import Any


_ALLOWED_COMMAND = {
    "torch": torch,
    "abs": torch.abs,
}


@dataclass(frozen=True)
class CommandConstraint:
    target: str
    operator: str
    direction: str | None
    referenced_terms: frozenset[str]
    compiled_expression: Any


class CommandConstraintSet:

    def __init__(
        self,
        term_names: set[str],
        constraints: dict[str, dict[str, Any]] | None,
    ) -> None:
        
        self.term_names = term_names
        self.constraints = self._build(constraints)


    def _build(
        self,
        constraints: dict[str, dict[str, Any]] | None,
    ) -> tuple[CommandConstraint, ...]:
        
        if constraints is None:
            return ()

        built: list[CommandConstraint] = []
        for target, config in constraints.items():
            if target not in self.term_names:
                raise ValueError(
                    f"Constraint target '{target}' is not a command term."
                )
            if not isinstance(config, dict):
                raise TypeError(
                    f"Config of constraint '{target}' must be a dict."
                )

            operator = config.get("operator")
            expression = config.get("expression")
            if operator is None or expression is None:
                raise ValueError(
                    f"'operator' or 'expression' is missing for "
                    f"constraint '{target}'."
                )
            if operator not in {"<=", ">=", "<", ">", "!="}:
                raise ValueError(
                    f"Unsupported operator '{operator}' for "
                    f"constraint '{target}'."
                )
            direction = config.get("direction")
            if operator == "!=":
                if direction is None:
                    direction = "positive"
                elif direction not in {"positive", "negative"}:
                    raise ValueError(
                        f"'direction' of constraint '{target}' must be "
                        "'positive' or 'negative'."
                    )
            elif direction is not None:
                raise ValueError(
                    f"'direction' is only supported by the '!=' operator, "
                    f"got operator '{operator}' for constraint '{target}'."
                )
            if not isinstance(expression, str):
                raise TypeError(
                    f"'expression' of constraint '{target}' must be a string."
                )

            referenced_terms = frozenset(
                re.findall(r"\{([A-Za-z_]\w*)\}", expression)
            )
            unknown_terms = referenced_terms - self.term_names
            if unknown_terms:
                raise ValueError(
                    f"Constraint '{target}' references unknown command "
                    f"term(s): {sorted(unknown_terms)}."
                )
            compiled_expression = compile(
                re.sub(
                    r"\{([A-Za-z_]\w*)\}",
                    lambda match: f"commands[{match.group(1)!r}]",
                    expression,
                ),
                f"<command constraint: {target}>",
                "eval",
            )
            built.append(CommandConstraint(
                target=target,
                operator=operator,
                direction=direction,
                referenced_terms=referenced_terms,
                compiled_expression=compiled_expression,
            ))

        return tuple(built)


    def validate_groups(self, term_groups: dict[str, str]) -> None:

        for constraint in self.constraints:

            if constraint.target not in term_groups:
                continue

            target_group = term_groups[constraint.target]

            for referenced_term in constraint.referenced_terms:

                if term_groups.get(referenced_term) != target_group:
                    raise ValueError(
                        f"Constraint '{constraint.target}' crosses "
                        "curriculum groups."
                    )


    def correct(
        self,
        commands: dict[str, torch.Tensor],
        target_names: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:

        checked_commands = dict(commands)

        for constraint in self.constraints:

            if (
                target_names is not None
                and constraint.target not in target_names
            ):
                continue
            
            command = checked_commands[constraint.target]
            expression_commands = {
                name: checked_commands[name]
                for name in constraint.referenced_terms
            }

            try:
                boundary = eval(
                    constraint.compiled_expression,
                    {"__builtins__": {}},
                    {
                        "commands": expression_commands,
                        **_ALLOWED_COMMAND,
                    },
                )
                boundary = torch.as_tensor(
                    boundary,
                    dtype=command.dtype,
                    device=command.device,
                )

            except Exception as error:
                raise ValueError(
                    f"Error evaluating constraint "
                    f"'{constraint.target}': {error}"
                ) from error

            if constraint.operator == "<=":
                checked_command = torch.minimum(command, boundary)
            elif constraint.operator == ">=":
                checked_command = torch.maximum(command, boundary)
            elif constraint.operator == "<":
                strict_boundary = torch.nextafter(
                    boundary,
                    torch.full_like(boundary, -torch.inf),
                )
                checked_command = torch.minimum(command, strict_boundary)
            elif constraint.operator == ">":
                strict_boundary = torch.nextafter(
                    boundary,
                    torch.full_like(boundary, torch.inf),
                )
                checked_command = torch.maximum(command, strict_boundary)
            else:
                direction = (
                    torch.inf
                    if constraint.direction == "positive"
                    else -torch.inf
                )
                replacement = torch.nextafter(
                    boundary,
                    torch.full_like(boundary, direction),
                )
                checked_command = torch.where(
                    command != boundary,
                    command,
                    replacement,
                )

            checked_commands[constraint.target] = checked_command

        return checked_commands
    

    def filter(
        self,
        command_starts: dict[str, torch.Tensor],
        cell_interval: dict[str, torch.Tensor],
        target_names: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:

        filtered_starts = dict(command_starts)

        missing_intervals = set(filtered_starts) - cell_interval.keys()
        if missing_intervals:
            raise ValueError(
                "Missing cell intervals for command term(s): "
                f"{sorted(missing_intervals)}."
            )

        for constraint in self.constraints:
            if (
                target_names is not None
                and constraint.target not in target_names
            ):
                continue

            target_start = filtered_starts[constraint.target]
            target_end = (
                target_start
                + cell_interval[constraint.target]
            )
            boundary_min, boundary_max, boundary_is_nan = (
                self._expression_bounds(
                    constraint,
                    filtered_starts,
                    cell_interval,
                    target_start,
                )
            )

            if constraint.operator in {"<=", "<"}:
                valid = target_end <= boundary_min
            elif constraint.operator == ">=":
                valid = target_start >= boundary_max
            elif constraint.operator == ">":
                valid = target_start > boundary_max
            else:
                valid = (
                    (target_end <= boundary_min)
                    | (target_start > boundary_max)
                    | boundary_is_nan
                )

            valid_cells = valid.all(dim=-1)
            filtered_starts = {
                name: starts[valid_cells]
                for name, starts in filtered_starts.items()
            }

        return filtered_starts



    def _expression_bounds(
        self,
        constraint: CommandConstraint,
        command_starts: dict[str, torch.Tensor],
        cell_interval: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        referenced_terms = tuple(sorted(constraint.referenced_terms))
        referenced_dim = sum(
            command_starts[name].shape[-1]
            for name in referenced_terms
        )
        if referenced_dim > 20:
            raise ValueError(
                f"Constraint '{constraint.target}' references "
                f"{referenced_dim} scalar dimensions; interval checking "
                "supports at most 20."
            )

        boundaries: list[torch.Tensor] = []
        for corner_id in range(1 << referenced_dim):
            expression_commands: dict[str, torch.Tensor] = {}
            dimension_offset = 0
            for name in referenced_terms:
                starts = command_starts[name]
                interval = cell_interval[name]
                use_end = torch.tensor(
                    [
                        bool(corner_id & (1 << (dimension_offset + index)))
                        for index in range(starts.shape[-1])
                    ],
                    dtype=torch.bool,
                    device=starts.device,
                )
                expression_commands[name] = torch.where(
                    use_end,
                    starts + interval,
                    starts,
                )
                dimension_offset += starts.shape[-1]

            try:
                boundary = eval(
                    constraint.compiled_expression,
                    {"__builtins__": {}},
                    {
                        "commands": expression_commands,
                        **_ALLOWED_COMMAND,
                    },
                )
                boundary = torch.as_tensor(
                    boundary,
                    dtype=target.dtype,
                    device=target.device,
                )
                boundaries.append(torch.broadcast_to(boundary, target.shape))
            except Exception as error:
                raise ValueError(
                    f"Error evaluating constraint "
                    f"'{constraint.target}': {error}"
                ) from error

        stacked = torch.stack(boundaries)
        boundary_is_nan = stacked.isnan().all(dim=0)
        boundary_min = stacked.nan_to_num(nan=torch.inf).amin(dim=0)
        boundary_max = stacked.nan_to_num(nan=-torch.inf).amax(dim=0)

        return boundary_min, boundary_max, boundary_is_nan
