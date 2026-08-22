import re
import torch
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandConstraint:
    target: str
    operator: str
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
            if operator not in {"<=", ">="}:
                raise ValueError(
                    f"Unsupported operator '{operator}' for "
                    f"constraint '{target}'."
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
                        "abs": torch.abs,
                        "torch": torch,
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

            checked_commands[constraint.target] = (
                torch.minimum(command, boundary)
                if constraint.operator == "<="
                else torch.maximum(command, boundary)
            )

        return checked_commands
