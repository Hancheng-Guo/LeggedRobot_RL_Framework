from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module

from envs.simulators.base import BaseSimulator


class SimulatorRegistry(Mapping[str, type[BaseSimulator]]):
    """Resolve simulator implementations only when they are selected."""

    def __init__(
        self,
        entries: Mapping[str, tuple[str, str]]
    ) -> None:
        
        self._entries = dict(entries)
        self._resolved: dict[str, type[BaseSimulator]] = {}


    def __getitem__(
        self,
        name: str
    ) -> type[BaseSimulator]:
        
        try:
            module_name, class_name = self._entries[name]
        except KeyError:
            raise KeyError(name) from None

        simulator_type = self._resolved.get(name)
        if simulator_type is None:
            module = import_module(module_name)
            candidate = getattr(module, class_name)
            if not isinstance(candidate, type) or not issubclass(
                candidate,
                BaseSimulator,
            ):
                raise TypeError(
                    f"Registered simulator {name!r} does not resolve to a "
                    "BaseSimulator subclass."
                )
            simulator_type = candidate
            self._resolved[name] = simulator_type
        return simulator_type


    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)


    def __len__(self) -> int:
        return len(self._entries)


SIM_TYPE_MAP: Mapping[str, type[BaseSimulator]] = SimulatorRegistry({
    "mujoco": ("envs.simulators.mujoco", "MujocoSimulator"),
    "isaac_sim": ("envs.simulators.isaac_sim", "IsaacSimSimulator"),
})
