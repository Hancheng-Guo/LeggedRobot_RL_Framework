from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from app.utils.context import RuntimeContext
from rl.algorithms.ppo import PPO
from rl.policies.configurable_actor_critic import ConfigurableActorCritic
from rl.policies.modules.registry import build_module
from rl.policies.modules.recurrent import StatefulGRU
from rl.policies.registry import POLICY_TYPE_MAP
from rl.utils.storage import RolloutStorage
from utils.component import Component, ComponentInfo


ACTOR_CONFIG: list[dict[str, Any]] = [
    {"type": "linear", "in_features": 3, "out_features": 8},
    {"type": "tanh"},
    {"type": "linear", "in_features": 8, "out_features": "action_dim"},
]

CRITIC_CONFIG = [
    {"type": "linear", "in_features": 3, "out_features": 8},
    {"type": "tanh"},
    {"type": "linear", "in_features": 8, "out_features": 1},
]

RECURRENT_ACTOR_CONFIG: list[dict[str, Any]] = [
    {"type": "gru", "input_size": 3, "hidden_size": 8},
    {"type": "linear", "in_features": 8, "out_features": "action_dim"},
]


def make_component(policy_config: Path | None = None) -> Component:
    policy = (
        None
        if policy_config is None
        else ComponentInfo(
            type="actor_critic",
            config=policy_config,
        )
    )
    return Component(
        runner=None,
        algorithm=None,
        policy=policy,
        environment=None,
        simulator=None,
        task=None,
    )


def test_configurable_actor_critic_builds_and_evaluates(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=ACTOR_CONFIG,
        critic=CRITIC_CONFIG,
        distribution={
            "type": "diagonal_gaussian",
            "initial_std": 0.5,
        },
    )

    obs = torch.zeros(
        4,
        3,
        device=runtime_context.device,
        dtype=runtime_context.dtype,
    )
    output = policy.act(obs=obs, deterministic=True)
    evaluation = policy.evaluate_actions(obs=obs, actions=output.action)

    assert output.action.shape == (4, 2)
    assert output.log_prob.shape == (4,)
    assert output.value.shape == (4,)
    assert evaluation.log_prob.shape == (4,)
    assert evaluation.entropy.shape == (4,)
    assert evaluation.value.shape == (4,)
    assert output.action.device == runtime_context.device
    assert output.action.dtype == runtime_context.dtype


def test_configurable_actor_critic_resolves_obs_dim(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=5,
        action_dim=2,
        actor=[
            {"type": "linear", "in_features": "obs_dim", "out_features": 8},
            {"type": "linear", "in_features": 8, "out_features": "action_dim"},
        ],
        critic=[
            {"type": "linear", "in_features": "obs_dim", "out_features": 8},
            {"type": "linear", "in_features": 8, "out_features": 1},
        ],
        distribution={"type": "diagonal_gaussian"},
    )

    assert policy.actor[0].in_features == 5
    assert policy.critic[0].in_features == 5


def test_stateful_gru_rejects_online_batch_size_change() -> None:
    module = StatefulGRU(input_size=3, hidden_size=4)
    module(torch.zeros(2, 3))

    with pytest.raises(ValueError, match="batch size"):
        module(torch.zeros(3, 3))


def test_policy_registry_contains_actor_critic() -> None:
    assert (
        POLICY_TYPE_MAP["actor_critic"]
        is ConfigurableActorCritic
    )


def test_recurrent_actor_owns_and_resets_hidden_state(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=RECURRENT_ACTOR_CONFIG,
        critic=CRITIC_CONFIG,
        distribution={
            "type": "diagonal_gaussian",
            "initial_std": 0.5,
        },
    )
    obs = torch.ones(
        2,
        3,
        device=runtime_context.device,
        dtype=runtime_context.dtype,
    )

    assert policy.is_recurrent is True
    initial_state = policy.get_recurrent_state(batch_size=2)
    policy.act(obs)
    advanced_state = policy.get_recurrent_state()
    state_name = next(iter(advanced_state))
    assert not torch.equal(advanced_state[state_name], initial_state[state_name])

    policy.reset_recurrent_state(torch.tensor([0]))
    reset_state = policy.get_recurrent_state()
    assert torch.count_nonzero(reset_state[state_name][0]) == 0
    assert torch.count_nonzero(reset_state[state_name][1]) > 0


def test_recurrent_sequence_evaluation_preserves_time_gradients(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=RECURRENT_ACTOR_CONFIG,
        critic=CRITIC_CONFIG,
        distribution={
            "type": "diagonal_gaussian",
            "initial_std": 0.5,
        },
    )
    initial_state = policy.get_recurrent_state(batch_size=2)
    obs = torch.randn(
        4,
        2,
        3,
        device=runtime_context.device,
        dtype=runtime_context.dtype,
    )
    actions = torch.zeros(4, 2, 2, device=runtime_context.device)
    reset_mask = torch.zeros(4, 2, dtype=torch.bool)
    reset_mask[2, 0] = True

    online_state_before = policy.get_recurrent_state()
    evaluation, final_state = policy.evaluate_recurrent_sequences(
        obs=obs,
        actions=actions,
        initial_state=initial_state,
        reset_mask=reset_mask,
    )
    loss = -evaluation.log_prob.mean()
    loss.backward()

    assert evaluation.log_prob.shape == (4, 2)
    assert set(final_state) == set(initial_state)
    online_state_after = policy.get_recurrent_state()
    for name in online_state_before:
        torch.testing.assert_close(
            online_state_after[name],
            online_state_before[name],
        )
    gru_parameter = next(policy.actor[0].parameters())
    assert gru_parameter.grad is not None


def test_ppo_evaluation_does_not_create_rollout_state(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=RECURRENT_ACTOR_CONFIG,
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    algorithm = PPO(context=runtime_context)
    algorithm.policy = policy
    algorithm.set_eval_mode()

    obs = torch.ones(
        2,
        3,
        device=runtime_context.device,
        dtype=runtime_context.dtype,
    )
    algorithm.act(obs, deterministic=True)

    assert not hasattr(algorithm, "_pending_recurrent_state")
    state_name = next(iter(policy.get_recurrent_state()))
    algorithm.reset_policy_state(torch.tensor([0]))
    state = policy.get_recurrent_state()
    assert torch.count_nonzero(state[state_name][0]) == 0
    assert torch.count_nonzero(state[state_name][1]) > 0


def test_ppo_updates_recurrent_actor_sequences(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=RECURRENT_ACTOR_CONFIG,
        critic=CRITIC_CONFIG,
        distribution={
            "type": "diagonal_gaussian",
            "initial_std": 0.5,
        },
    )
    algorithm = PPO(context=runtime_context)
    algorithm.policy = policy
    algorithm.storage = RolloutStorage(context=runtime_context)
    algorithm.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    algorithm.learning_rate = 3e-4
    algorithm.gamma = 0.99
    algorithm.gae_lambda = 0.95
    algorithm.clip_range = 0.2
    algorithm.entropy_coef = 0.01
    algorithm.value_coef = 0.5
    algorithm.max_grad_norm = 0.5
    algorithm.num_epochs = 1
    algorithm.num_mini_batches = 1

    obs = torch.zeros(
        2,
        3,
        device=runtime_context.device,
        dtype=runtime_context.dtype,
    )
    for step in range(3):
        output = algorithm.act(obs)
        terminated = torch.tensor(
            [step == 1, False],
            device=runtime_context.device,
        )
        next_obs = obs + 1.0
        algorithm.process_transition(
            obs=obs,
            policy_output=output,
            reward=torch.ones(2, device=runtime_context.device),
            terminated=terminated,
            truncated=torch.zeros(2, dtype=torch.bool),
            next_obs=next_obs,
        )
        obs = next_obs

    algorithm.compute_returns(last_obs=obs)
    info = algorithm.update()

    assert "rollout/loss" in info
    assert algorithm.storage.obs == []


def test_module_factory_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Invalid module type"):
        build_module({"type": "unknown"})


def test_module_factory_accepts_nested_params() -> None:
    module = build_module({
        "type": "linear",
        "params": {
            "in_features": 3,
            "out_features": 2,
        },
    })

    assert isinstance(module, torch.nn.Linear)
    assert module.in_features == 3
    assert module.out_features == 2


def test_ppo_builds_policy_before_optimizer(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "actor": ACTOR_CONFIG,
            "critic": CRITIC_CONFIG,
            "distribution": {
                "type": "diagonal_gaussian",
                "initial_std": 0.5,
            },
        }),
        encoding="utf-8",
    )
    algorithm = PPO(context=runtime_context)
    algorithm.config_update(
        component=make_component(config_path),
        obs_dim=3,
        action_dim=2,
        init_learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        num_epochs=2,
        num_mini_batches=1,
    )

    assert isinstance(algorithm.policy, ConfigurableActorCritic)
    assert algorithm.init_learning_rate == 3e-4
    assert algorithm.learning_rate == 3e-4
    assert algorithm.optimizer.param_groups[0]["lr"] == 3e-4
    assert len(algorithm.optimizer.param_groups[0]["params"]) > 0
