from importlib import import_module
from typing import Any

from .base import BaseCallback


_LAZY_EXPORTS = {
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
    "CALLBACK_TYPE_MAP": (
        "runners.callbacks.registry",
        "CALLBACK_TYPE_MAP",
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
    
    *_LAZY_EXPORTS,
)
