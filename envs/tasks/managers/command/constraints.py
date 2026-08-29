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


    def apply(
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
