from typing import Any
from abc import ABC, abstractmethod


class BaseCallback(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass


    def _on_train_begin(self) -> None:
        pass


    def _on_train_end(self) -> None:
        pass


    def _on_close(self) -> None:
        pass


    def _on_iteration_begin(self) -> None:
        pass


    def _on_iteration_end(self) -> None:
        pass


    def _on_step_begin(self) -> None:
        pass


    def _on_step_end(self) -> None:
        pass

