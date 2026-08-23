import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(rewards[-1])

    for step in range(rewards.shape[0] - 1, -1, -1):
        bootstrap_mask = (~terminated[step]).to(rewards.dtype)
        continuation_mask = (
            ~(terminated[step] | truncated[step])
        ).to(rewards.dtype)

        delta = (
            rewards[step]
            + gamma * next_values[step] * bootstrap_mask
            - values[step]
        )
        next_advantage = (
            delta
            + gamma * gae_lambda * continuation_mask * next_advantage
        )
        advantages[step] = next_advantage

    return advantages + values, advantages
