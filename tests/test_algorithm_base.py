from collections.abc import Mapping
from typing import Any

import torch

from app.utils.context import RuntimeContext
from rl.algorithms.base import (
    OnPolicyAlgorithm,
    PolicyOutput,
)
from rl.algorithms.ppo import PPO
from rl.policies.base import BasePolicy, PolicyActionOutput, PolicyEvaluation
from rl.utils.storage import RolloutStorage
from utils.component import Component


class FakePPOPolicy(BasePolicy):

    def __init__(self, context: RuntimeContext) -> None:
        super().__init__(context=context)
        self.action_mean = torch.nn.Parameter(torch.zeros(1))
        self.value_bias = torch.nn.Parameter(torch.zeros(1))
        self.closed = False


    def config_update(
        self,
        component: Component,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None


    def get_action_distribution(
        self,
        obs: torch.Tensor,
    ) -> torch.distributions.Normal:
        mean = self.action_mean.expand(obs.shape[0], 1)
        return torch.distributions.Normal(mean, torch.ones_like(mean))


    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_bias.expand(obs.shape[0])


    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyActionOutput:
        distribution = self.get_action_distribution(obs)
        action = distribution.mean if deterministic else distribution.sample()
        return PolicyActionOutput(
            action=action,
            log_prob=distribution.log_prob(action).sum(dim=-1),
            value=self.predict_values(obs),
        )


    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> PolicyEvaluation:
        distribution = self.get_action_distribution(obs)
        return PolicyEvaluation(
            log_prob=distribution.log_prob(actions).sum(dim=-1),
            entropy=distribution.entropy().sum(dim=-1),
            value=self.predict_values(obs),
        )


    def set_train_mode(self) -> None:
        self.train()


    def set_eval_mode(self) -> None:
        self.eval()


    def close(self) -> None:
        self.closed = True


class TrackingRolloutStorage(RolloutStorage):

    def __init__(self, context: RuntimeContext) -> None:
        super().__init__(context=context)
        self.closed = False


    def close(self) -> None:
        super().close()
        self.closed = True


class DummyOnPolicyAlgorithm(OnPolicyAlgorithm):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:
        
        super().__init__(context)
        self.last_deterministic: bool | None = None


    def config_update(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return None


    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        
        self.last_deterministic = deterministic
        return PolicyOutput(
            action=obs,
            log_prob=torch.zeros_like(obs),
            value=torch.zeros_like(obs),
        )


    def process_transition(
        self,
        obs: torch.Tensor,
        policy_output: PolicyOutput,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: torch.Tensor,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        return None


    def compute_returns(
        self,
        last_obs: torch.Tensor,
    ) -> None:
        return None


    def update(self) -> dict[str, float]:
        return {"loss": 0.0}


    def close(self) -> None:
        return None


def test_algorithm_defaults_to_stochastic_action(
    runtime_context: RuntimeContext,
) -> None:

    algorithm = DummyOnPolicyAlgorithm(runtime_context)
    algorithm.act(torch.ones(1))

    assert algorithm.last_deterministic is False


def test_algorithm_controls_policy_mode(
    runtime_context: RuntimeContext,
) -> None:

    algorithm = DummyOnPolicyAlgorithm(runtime_context)
    algorithm.policy = FakePPOPolicy(context=runtime_context)

    algorithm.set_eval_mode()
    assert algorithm.policy.training is False

    algorithm.set_train_mode()
    assert algorithm.policy.training is True


def test_partial_ppo_remains_instantiable(
    runtime_context: RuntimeContext,
) -> None:

    algorithm = PPO(runtime_context)

    assert isinstance(algorithm, OnPolicyAlgorithm)


def test_ppo_processes_transition_and_returns(
    runtime_context: RuntimeContext,
) -> None:
    algorithm = PPO(runtime_context)
    policy = FakePPOPolicy(context=runtime_context)
    storage = RolloutStorage(context=runtime_context)
    algorithm.policy = policy
    algorithm.storage = storage
    algorithm.gamma = 0.99
    algorithm.gae_lambda = 0.95

    obs = torch.zeros(2, 3)
    output = algorithm.act(obs)
    algorithm.process_transition(
        obs=obs,
        policy_output=output,
        reward=torch.ones(2),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        next_obs=obs + 1.0,
    )
    algorithm.compute_returns(last_obs=obs)

    assert torch.equal(storage.actions[0], output.action)
    assert storage.returns is not None
    assert storage.advantages is not None


def test_ppo_update_optimizes_and_reports_metrics(
    runtime_context: RuntimeContext,
) -> None:
    algorithm = PPO(runtime_context)
    policy = FakePPOPolicy(context=runtime_context)
    storage = RolloutStorage(context=runtime_context)
    algorithm.policy = policy
    algorithm.storage = storage
    algorithm.optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    algorithm.clip_range = 0.2
    algorithm.entropy_coef = 0.01
    algorithm.value_coef = 0.5
    algorithm.max_grad_norm = 0.5
    algorithm.num_epochs = 2
    algorithm.num_mini_batches = 1
    algorithm.gamma = 0.99
    algorithm.gae_lambda = 0.95

    obs = torch.zeros(4, 3)
    actions = torch.full((4, 1), 0.5)
    old_distribution = policy.get_action_distribution(obs)
    storage.add(
        obs=obs,
        action=actions,
        log_prob=old_distribution.log_prob(actions).sum(dim=-1),
        value=policy.predict_values(obs),
        reward=torch.ones(4),
        terminated=torch.zeros(4, dtype=torch.bool),
        truncated=torch.zeros(4, dtype=torch.bool),
        next_obs=obs,
    )
    algorithm.compute_returns(last_obs=obs)

    info = algorithm.update()

    assert storage.obs == []
    assert set(info) == {
        "rollout/loss",
        "rollout/policy_loss",
        "rollout/value_loss",
        "rollout/entropy",
        "rollout/approx_kl",
        "rollout/clip_fraction",
        "rollout/grad_norm",
    }


def test_ppo_closes_owned_components(
    runtime_context: RuntimeContext,
) -> None:
    
    algorithm = PPO(runtime_context)
    storage = TrackingRolloutStorage(context=runtime_context)
    policy = FakePPOPolicy(context=runtime_context)
    algorithm.storage = storage
    algorithm.policy = policy

    algorithm.close()

    assert storage.closed is True
    assert policy.closed is True
    assert not hasattr(algorithm, "storage")
    assert not hasattr(algorithm, "policy")

    algorithm.close()
