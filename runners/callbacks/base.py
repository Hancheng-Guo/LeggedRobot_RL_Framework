from abc import ABC, abstractmethod


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
