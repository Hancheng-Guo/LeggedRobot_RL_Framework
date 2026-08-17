from envs.tasks.base import BaseTaskLogic
from envs.tasks.locomotion import LocomotionTaskLogic


TASK_TYPE_MAP: dict[str, type[BaseTaskLogic] | type[LocomotionTaskLogic]] = {
    "base": BaseTaskLogic,
    "locomotion": LocomotionTaskLogic,
}
