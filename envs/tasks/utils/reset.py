"""One reset snapshot type for terms, managers and complete task transactions."""
from __future__ import annotations

import torch
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .checkpoint import state_field_paths


@dataclass
class ResetSnapshot:
    """Flat captured state, restored in place without rebuilding runtime objects."""

    tensors: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    attributes: list[tuple[Any, str, Any]] = field(default_factory=list)
    containers: list[tuple[list | dict, list | dict]] = field(default_factory=list)
    generators: list[tuple[torch.Generator, torch.Tensor]] = field(default_factory=list)


    def extend(self, snapshot: ResetSnapshot) -> None:
        """Include another owner's captured data without nesting snapshot types."""
        self.tensors.extend(snapshot.tensors)
        self.attributes.extend(snapshot.attributes)
        self.containers.extend(snapshot.containers)
        self.generators.extend(snapshot.generators)


    def restore(self) -> None:
        """Attempt all restoration work, including RNGs, before reporting errors."""

        errors = []

        def attempt(operation, *args):
            try:
                operation(*args)
            except Exception as error:
                errors.append(error)

        with torch.no_grad():
            for target, saved in self.tensors:
                attempt(target.copy_, saved)
        for target, saved in self.containers:
            if isinstance(target, list):
                attempt(target.__setitem__, slice(None), saved)
            else:
                attempt(target.clear)
                attempt(target.update, saved)
        for owner, name, value in self.attributes:
            attempt(setattr, owner, name, value)
        for generator, state in self.generators:
            attempt(generator.set_state, state)
        if errors:
            raise ExceptionGroup("Reset snapshot restoration failed", errors)


class ResetStateOwner:
    """Declare this object's rollback fields and explicitly owned state objects."""

    RESET_STATE_FIELDS: ClassVar[tuple[str, ...]] = ()
    RESET_STATE_CHILDREN: ClassVar[tuple[str, ...]] = ()


    def _reset_state_children(self) -> Iterable[ResetStateOwner]:
        names = dict.fromkeys(
            name for cls in reversed(type(self).__mro__)
            for name in cls.__dict__.get("RESET_STATE_CHILDREN", ())
        )
        return (getattr(self, name) for name in names)


    def capture_reset_state(self) -> ResetSnapshot:
        names = dict.fromkeys(state_field_paths(
            self, "STATE_FIELDS", "RESET_STATE_FIELDS").values()
        )
        snapshot = _capture_fields(self, names)
        for child in self._reset_state_children():
            snapshot.extend(child.capture_reset_state())
        return snapshot


class ResetStateManager(ResetStateOwner):

    terms: Any

    def _reset_state_children(self) -> Iterable[ResetStateOwner]:
        return (*super()._reset_state_children(), *self.terms.values())


def capture_task_reset_state(
    managers: Iterable[ResetStateOwner]
) -> ResetSnapshot:
    """Capture task managers and shared training RNGs in the same snapshot type."""

    snapshot = ResetSnapshot(
        generators=[(torch.default_generator, torch.get_rng_state().clone())]
    )
    if torch.cuda.is_initialized():
        snapshot.generators.extend(
            (generator, generator.get_state().clone())
            for generator in torch.cuda.default_generators
        )
    for manager in managers:
        snapshot.extend(manager.capture_reset_state())
    return snapshot


def _capture_fields(
    owner: Any,
    names: Iterable[str],
) -> ResetSnapshot:

    snapshot = ResetSnapshot()
    seen: set[int] = set()

    def collect(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            snapshot.tensors.append((value, value.detach().clone()))
        elif isinstance(value, torch.Generator):
            snapshot.generators.append((value, value.get_state().clone()))
        elif isinstance(value, Mapping):
            if isinstance(value, dict):
                snapshot.containers.append((value, value.copy()))
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            if isinstance(value, list):
                snapshot.containers.append((value, value.copy()))
            for item in value:
                collect(item)

    for path in names:
        target = owner
        parts = path.split(".")
        for name in parts[:-1]:
            target = getattr(target, name)
        value = getattr(target, parts[-1])
        collect(value)
        snapshot.attributes.append((target, parts[-1], value))
    return snapshot
