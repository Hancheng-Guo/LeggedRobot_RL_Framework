import torch

from rl.utils.gae import compute_gae
from rl.utils.storage import RolloutStorage


def test_compute_gae_bootstraps_truncation_without_crossing_reset():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[0.0], [10.0]])
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = torch.tensor([[False], [True]])

    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([[13.0], [12.0]]))
    torch.testing.assert_close(returns, advantages)


def test_rollout_storage_generates_typed_mini_batches(runtime_context):
    storage = RolloutStorage(context=runtime_context)

    for step in range(2):
        obs = torch.tensor([
            [float(step * 2)],
            [float(step * 2 + 1)],
        ])
        storage.add(
            obs=obs,
            action=obs,
            log_prob=torch.zeros(2),
            value=torch.zeros(2),
            reward=torch.ones(2),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            next_obs=obs + 1.0,
        )

    storage.set_returns(
        returns=torch.ones(2, 2),
        advantages=torch.ones(2, 2),
    )
    batches = list(storage.mini_batches(num_mini_batches=2))

    assert len(batches) == 2
    assert sum(batch.obs.shape[0] for batch in batches) == 4
    actual_obs = torch.cat([
        batch.obs for batch in batches
    ]).flatten().sort().values
    torch.testing.assert_close(
        actual_obs,
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )


def test_rollout_storage_clear_resets_rollout(runtime_context):
    storage = RolloutStorage(context=runtime_context)
    obs = torch.zeros(1, 1)
    storage.add(
        obs=obs,
        action=obs,
        log_prob=torch.zeros(1),
        value=torch.zeros(1),
        reward=torch.zeros(1),
        terminated=torch.zeros(1, dtype=torch.bool),
        truncated=torch.zeros(1, dtype=torch.bool),
        next_obs=obs,
    )
    storage.set_returns(
        returns=torch.zeros(1, 1),
        advantages=torch.zeros(1, 1),
    )

    storage.clear()

    assert storage.obs == []
    assert storage.returns is None
    assert storage.advantages is None
