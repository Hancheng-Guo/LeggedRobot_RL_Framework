import torch

from envs.simulators.utils.context import ModelContext
from envs.tasks.utils.context import TaskContext
from envs.tasks.managers.reward.terms.registry import register_reward


@register_reward
def action_diff_l2(
    task_context: TaskContext,
    model_context: ModelContext,
) -> torch.Tensor:

    action_diff = task_context.action - task_context.last_action

    return torch.mean(action_diff.square(), dim=-1)