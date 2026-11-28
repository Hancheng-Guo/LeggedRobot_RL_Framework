"""Combined entry points for owners needing both rollback and persistent state."""
from .reset import ResetStateOwner, ResetStateManager

from utils import CheckpointStateOwner, CheckpointStateManager


class StateOwner(ResetStateOwner, CheckpointStateOwner):
    """Declare shared state once while retaining separate restoration semantics."""

    def __init__(
        self,
        *,
        name: str,
    ) -> None:

        if not isinstance(name, str) or not name:
            raise ValueError("State owner name must be a non-empty string.")
        self.name = name


    def config_update(self, **kwargs) -> None:
        """Refresh runtime dependencies; subclasses consume only what they need."""
        pass


    def set_training_mode(self, training: bool) -> None:
        """Override when behavior depends on training or evaluation mode."""
        pass


class StateManager(ResetStateManager, CheckpointStateManager):
    """Compose term state for both reset transactions and training checkpoints."""
