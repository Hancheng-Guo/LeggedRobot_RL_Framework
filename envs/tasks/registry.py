from envs.tasks.locomotion import LocomotionTaskLogic


TASK_TYPE_MAP: dict[str, type[LocomotionTaskLogic]] = {
    "locomotion": LocomotionTaskLogic,
}
