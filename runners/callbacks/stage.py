from runners.callbacks.base import BaseCallback
from runners.base import BaseRunner


class StageCallback(BaseCallback):

    def __init__(
        self,
        condition: dict,
        **kwargs,
    ) -> None:
        self.condition = condition
        self.runner = None
        self.stop_training = False


    def set_runner(
        self,
        runner: BaseRunner,
    ) -> None:
        self.runner = runner