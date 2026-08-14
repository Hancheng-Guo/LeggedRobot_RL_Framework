from typing import Any
from abc import ABC, abstractmethod

from runners.base import BaseRunner


class BaseCallback(ABC):

    @abstractmethod
    def __init__(
        self,
        runner: BaseRunner,
        *args, **kwargs,
    ) -> None:
        pass


    def _on_train_start(self) -> None:
        pass


    def _on_train_end(self) -> None:
        pass


    def _on_iteration_start(self) -> None:
        pass


    def _on_iteration_end(self) -> None:
        pass


    def _on_step_start(self) -> None:
        pass


    def _on_step_end(self) -> None:
        pass

    
    def _on_close(self) -> None:
        pass
