"""Pass manager descriptions to CurriculumManager as LocomotionTaskLogic does."""
from envs.tasks.locomotion import LocomotionTaskLogic
from envs.tasks.managers.curriculum import CurriculumManager


def build_curriculum_manager(*, manager_configs, context, terms, **kwargs):
    return CurriculumManager(
        manager_configs=manager_configs, context=context, terms=terms,
        space_configs=LocomotionTaskLogic.get_curriculum_space_configs(manager_configs), **kwargs,
    )
