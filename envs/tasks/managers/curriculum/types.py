import torch
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CurriculumTermProvider(Protocol):

    def get_term(self, name: str) -> Any:
        ...


@runtime_checkable
class CommandCurriculumSampler(Protocol):

    @property
    def buffers(self) -> dict[str, Any]:
        ...

    def resample(
        self,
        space_names: set[str],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        ...

    def get_command(
        self,
        space_name: str,
        dimension: str,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
