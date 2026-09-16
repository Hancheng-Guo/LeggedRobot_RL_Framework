from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseCallback(ABC):

    @abstractmethod
    def __init__(
        self,
        runner,
        *args, **kwargs,
    ) -> None:
        self.runner = runner


    def _on_train_start(
        self,
        *args, **kwargs,
    ) -> bool:
        return True


    def _on_train_end(
        self,
        *args, **kwargs,
    ) -> bool:
        return True


    def _on_iteration_start(
        self,
        *args, **kwargs,
    ) -> bool:
        return True


    def _on_iteration_end(
        self,
        *args, **kwargs,
    ) -> bool:
        return True


    def _on_iteration_finalize(
        self,
        *args, **kwargs,
    ) -> bool:
        """Run after every callback has processed the iteration result."""
        return True


    def _on_step_start(
        self,
        *args, **kwargs,
    ) -> bool:
        return True


    def _on_step_end(
        self,
        *args, **kwargs,
    ) -> bool:
        return True

    
    def _on_close(
        self,
        *args, **kwargs,
    ) -> bool:
        return True


    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Return runtime state that must survive an interrupted training run."""
        return {}


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        """Restore runtime state after the callback has been initialized."""
        if state:
            raise ValueError(
                f"{type(self).__name__} does not define checkpoint state."
            )
