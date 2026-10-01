from .base import BaseCallback
from .progress_bar import ProgressBarCallback
from .checkpoint import CheckpointCallback
from .tensorboard import TensorboardCallback
from .early_stopping import EarlystoppingCallback
from .logging import LoggingCallback
from .keyboard_interrupt import KeyboardInterruptCallback
from .adaptive_learning_rate import AdaptiveLearningRateCallback


CALLBACK_TYPE_MAP: dict[str, type[BaseCallback]] = {
    "progress_bar": ProgressBarCallback,
    "checkpoint": CheckpointCallback,
    "tensorboard": TensorboardCallback,
    "early_stopping": EarlystoppingCallback,
    "logging": LoggingCallback,
    "keyboard_interrupt": KeyboardInterruptCallback,
    "adaptive_learning_rate": AdaptiveLearningRateCallback,
}
