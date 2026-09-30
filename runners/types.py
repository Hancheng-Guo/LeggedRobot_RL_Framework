"""Public runner lifecycle and checkpoint information."""

from dataclasses import dataclass
from enum import Enum, auto


class RestoreMode(Enum):
    RESUME = auto()
    ADVANCE_STAGE = auto()
    EVALUATE = auto()


class TrainStopReason(Enum):
    STAGE_COMPLETED = auto()
    MAX_ITERATIONS_REACHED = auto()
    CALLBACK_STOPPED = auto()


@dataclass(frozen=True)
class TrainResult:
    reason: TrainStopReason
    current_iteration: int
    stop_sources: tuple[str, ...] = ()
