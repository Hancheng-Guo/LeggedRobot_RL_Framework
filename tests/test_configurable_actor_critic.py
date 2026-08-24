from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from app.utils.context import RuntimeContext
from rl.algorithms.ppo import PPO
from rl.policies.configurable_actor_critic import ConfigurableActorCritic
from rl.policies.modules.registry import build_module
from rl.policies.registry import POLICY_TYPE_MAP
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


def test_policy_registry_contains_actor_critic() -> None:
    assert (
        POLICY_TYPE_MAP["actor_critic"]
        is ConfigurableActorCritic
    )


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
        action_dim=2,
        learning_rate=3e-4,
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
    assert algorithm.optimizer.param_groups[0]["lr"] == 3e-4
    assert len(algorithm.optimizer.param_groups[0]["params"]) > 0
