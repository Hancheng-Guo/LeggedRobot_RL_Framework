from .base import BaseTaskLogic
from .locomotion import LocomotionTaskLogic


TASK_TYPE_MAP: dict[str, type[BaseTaskLogic] | type[LocomotionTaskLogic]] = {
    "base": BaseTaskLogic,
    "locomotion": LocomotionTaskLogic,
}
