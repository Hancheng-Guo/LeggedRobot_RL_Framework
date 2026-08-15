from abc import ABC, abstractmethod


class BaseCallback(ABC):

    @abstractmethod
    def __init__(
        self,
        runner,
        *args, **kwargs,
    ) -> None:
        self.runner = runner
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
