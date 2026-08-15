from runners.callbacks.base import BaseCallback


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
        runner,
    ) -> None:
        self.runner = runner