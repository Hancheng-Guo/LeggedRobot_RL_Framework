import torch

from envs.tasks.managers.reward.terms.base import BaseRewardTerm
from envs.tasks.managers.reward.terms.registry import register_reward
from envs.tasks.utils.context import TaskContext


@register_reward
class ActionDiffL2(BaseRewardTerm):

    def compute(self, task_context: TaskContext) -> torch.Tensor:
        action_diff = task_context.action - task_context.last_action
        return torch.mean(action_diff.square(), dim=-1)
