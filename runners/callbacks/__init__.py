from importlib import import_module
from typing import TYPE_CHECKING, Any

from .base import BaseCallback

if TYPE_CHECKING:
    from .registry import CALLBACK_TYPE_MAP
    from .adaptive_learning_rate import AdaptiveLearningRateCallback, LearningRateAdjustment
    from .checkpoint import CheckpointCallback
    from .early_stopping import EarlystoppingCallback
    from .keyboard_interrupt import KeyboardInterruptCallback
    from .logging import LoggingCallback
    from .progress_bar import ProgressBarCallback
    from .stage import StageCallback
    from .tensorboard import TensorboardCallback


_LAZY_EXPORTS = {
    "CALLBACK_TYPE_MAP": (
        "runners.callbacks.registry",
        "CALLBACK_TYPE_MAP",
    ),
    "AdaptiveLearningRateCallback": (
        "runners.callbacks.adaptive_learning_rate",
        "AdaptiveLearningRateCallback",
    ),
    "LearningRateAdjustment": (
        "runners.callbacks.adaptive_learning_rate",
        "LearningRateAdjustment",
    ),
    "CheckpointCallback": (
        "runners.callbacks.checkpoint",
        "CheckpointCallback",
    ),
    "EarlystoppingCallback": (
        "runners.callbacks.early_stopping",
        "EarlystoppingCallback",
    ),
    "KeyboardInterruptCallback": (
        "runners.callbacks.keyboard_interrupt",
        "KeyboardInterruptCallback",
    ),
    "LoggingCallback": (
        "runners.callbacks.logging",
        "LoggingCallback",
    ),
    "ProgressBarCallback": (
        "runners.callbacks.progress_bar",
        "ProgressBarCallback",
    ),
    "StageCallback": (
        "runners.callbacks.stage",
        "StageCallback"
    ),
    "TensorboardCallback": (
        "runners.callbacks.tensorboard",
        "TensorboardCallback",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = (
    "BaseCallback",

    "CALLBACK_TYPE_MAP",

    "AdaptiveLearningRateCallback",
    "LearningRateAdjustment",

    "CheckpointCallback",

    "EarlystoppingCallback",

    "KeyboardInterruptCallback",

    "LoggingCallback",

    "ProgressBarCallback",

    "StageCallback",

    "TensorboardCallback",
)
