"""Validate and manage temporary commands during interactive playback."""

from collections.abc import Mapping

from .terms import BaseCommandTerm, PlaybackCommandSource, PlaybackCommandTerm


class PlaybackCommands:
    """Validate and wrap every configured command for interactive playback."""

    def __init__(
        self,
        terms: Mapping[str, BaseCommandTerm],
        source: PlaybackCommandSource,
        env_index: int,
        fixed_commands: Mapping[str, tuple[float, ...]] | None = None,
    ) -> None:
        
        fixed_commands = fixed_commands or {}
        source_names = source.values().keys()
        fixed_names = fixed_commands.keys()
        overlap = source_names & fixed_names

        if overlap:
            raise ValueError(f"Playback command(s) have two sources: {sorted(overlap)}.")
        unknown = (source_names | fixed_names) - terms.keys()
        if unknown:
            raise ValueError(f"Playback config has unknown command(s): {sorted(unknown)}.")

        missing = terms.keys() - source_names - fixed_names
        if missing:
            raise ValueError(f"Playback config is missing command(s): {sorted(missing)}.")

        non_scalar = [name for name in source_names if terms[name].dim != 1]
        if non_scalar:
            raise ValueError(
                f"ROS playback requires scalar command(s): {sorted(non_scalar)}."
            )

        wrong_size = [
            name for name, value in fixed_commands.items()
            if len(value) != terms[name].dim
        ]
        if wrong_size:
            raise ValueError(f"Playback fixed command dimension mismatch: {sorted(wrong_size)}.")

        self.original_terms = dict(terms)
        self.terms = {
            command_name: PlaybackCommandTerm(
                term, source, command_name, env_index,
                fixed_value=fixed_commands.get(command_name),
            )
            for command_name, term in terms.items()
        }
        self.env_index = env_index
