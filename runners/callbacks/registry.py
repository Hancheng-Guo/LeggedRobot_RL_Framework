from runners.callbacks.base import BaseCallback
from runners.callbacks.progress_bar import ProgressBarCallback
from runners.callbacks.checkpoint import CheckpointCallback
from runners.callbacks.tensorboard import TensorboardCallback
from runners.callbacks.early_stopping import EarlystoppingCallback
from runners.callbacks.logging import LoggingCallback


CALLBACK_TYPE_MAP: dict[str, type[BaseCallback]] = {
    "progress_bar": ProgressBarCallback,
    "checkpoint": CheckpointCallback,
    "tensorboard": TensorboardCallback,
    "early_stopping": EarlystoppingCallback,
    "logging": LoggingCallback,
}