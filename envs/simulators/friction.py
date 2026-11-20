"""Physical interface for task-owned sliding friction randomization."""
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from utils import RuntimeContext


@runtime_checkable
class FrictionBackend(Protocol):

    context: RuntimeContext
    num_envs: int

    def prepare_friction(self) -> None: ...

    def capture_friction(self, env_ids: Sequence[int]) -> object: ...

    def apply_friction(
        self, env_ids: Sequence[int], factors: Sequence[float],
    ) -> None: ...

    def restore_friction(self, snapshot: object) -> None: ...
