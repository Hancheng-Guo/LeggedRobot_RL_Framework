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


def test_configurable_actor_critic_resolves_action_shape(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=[
            {"type": "linear", "in_features": "obs.shape",
             "out_features": "action.shape"},
        ],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )

    assert policy.actor[0].out_features == 2


def test_policy_rejects_invalid_actor_and_critic_output_features(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)

    with pytest.raises(ValueError, match="Actor output features"):
        policy.config_update(
            component=make_component(), obs_dim=3, action_dim=2,
            actor=[{"type": "linear", "in_features": 3,
                    "out_features": 3}],
            critic=CRITIC_CONFIG,
            distribution={"type": "diagonal_gaussian"},
        )
    with pytest.raises(ValueError, match="Critic output features"):
        policy.config_update(
            component=make_component(), obs_dim=3, action_dim=2,
            actor=ACTOR_CONFIG,
            critic=[{"type": "linear", "in_features": 3,
                     "out_features": 2}],
            distribution={"type": "diagonal_gaussian"},
        )


def test_failed_distribution_update_preserves_existing_networks(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=ACTOR_CONFIG, critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    previous_actor = policy.actor
    previous_critic = policy.critic
    previous_distribution = policy.action_distribution

    with pytest.raises(ValueError, match="Invalid action distribution"):
        policy.config_update(
            component=make_component(), obs_dim=3, action_dim=2,
            actor=ACTOR_CONFIG, critic=CRITIC_CONFIG,
            distribution={"type": "missing"},
        )

    assert policy.actor is previous_actor
    assert policy.critic is previous_critic
    assert policy.action_distribution is previous_distribution


def test_network_rejects_inconsistent_module_input_features(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)

    with pytest.raises(ValueError, match="configured inputs provide 3"):
        policy.config_update(
            component=make_component(),
            obs_dim=3,
            action_dim=2,
            actor=[
                {"type": "linear", "in_features": 4,
                 "out_features": "action_dim"},
            ],
            critic=CRITIC_CONFIG,
            distribution={"type": "diagonal_gaussian"},
        )


def test_named_modules_can_inherit_parameters_between_stages(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    actor = [
        {"name": "encoder", "type": "linear", "in_features": 3,
         "out_features": 8},
        {"name": "activation", "type": "tanh"},
        {"name": "head", "type": "linear", "in_features": 8,
         "out_features": "action_dim"},
    ]
    critic = [
        {"name": "encoder", "type": "linear", "in_features": 3,
         "out_features": 8},
        {"name": "head", "type": "linear", "in_features": 8,
         "out_features": 1},
    ]
    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=actor, critic=critic,
        distribution={"type": "diagonal_gaussian"},
    )
    with torch.no_grad():
        policy.actor.encoder.weight.fill_(1.25)
        policy.critic.encoder.weight.fill_(2.5)

    inherited_actor = [dict(module) for module in actor]
    inherited_critic = [dict(module) for module in critic]
    inherited_actor[0]["inherit"] = True
    inherited_critic[0]["inherit"] = True
    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=inherited_actor, critic=inherited_critic,
        distribution={"type": "diagonal_gaussian"},
    )

    torch.testing.assert_close(
        policy.actor.encoder.weight,
        torch.full_like(policy.actor.encoder.weight, 1.25),
    )
    torch.testing.assert_close(
        policy.critic.encoder.weight,
        torch.full_like(policy.critic.encoder.weight, 2.5),
    )


def test_inherited_module_rejects_shape_changes(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=ACTOR_CONFIG, critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    changed_actor = [dict(module) for module in ACTOR_CONFIG]
    changed_actor[0].update({"inherit": True, "out_features": 9})
    changed_actor[2]["in_features"] = 9

    with pytest.raises(ValueError, match="parameter shapes differ"):
        policy.config_update(
            component=make_component(), obs_dim=3, action_dim=2,
            actor=changed_actor, critic=CRITIC_CONFIG,
            distribution={"type": "diagonal_gaussian"},
        )


def test_stage_can_reuse_named_composite_as_a_module_type(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    stage1_actor = [
        {
            "name": "custom1",
            "type": "sequential",
            "modules": [
                {"type": "linear", "in_features": 3, "out_features": 8},
                {"type": "elu"},
            ],
        },
        {"type": "linear", "in_features": 8,
         "out_features": "action_dim"},
    ]
    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=stage1_actor, critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    custom1 = policy.actor.custom1
    with torch.no_grad():
        custom1[0].weight.fill_(1.5)

    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=[
            {"type": "custom1"},
            {"type": "linear", "in_features": 8,
             "out_features": "action_dim"},
        ],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )

    torch.testing.assert_close(
        policy.actor.custom1[0].weight,
        torch.full_like(policy.actor.custom1[0].weight, 1.5),
    )


def test_composite_module_can_be_restored_from_artifact(
    runtime_context: RuntimeContext,
) -> None:
    first = ConfigurableActorCritic(context=runtime_context)
    first.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=[{
            "name": "custom1",
            "type": "sequential",
            "modules": [
                {"name": "memory", "type": "gru",
                 "input_size": 3, "hidden_size": 8},
                {"type": "linear", "in_features": 8,
                 "out_features": "action_dim"},
            ],
        }],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    with torch.no_grad():
        first.actor.custom1.memory.gru.weight_ih_l0.fill_(0.75)
    artifacts = first.export_module_artifacts()

    restored = ConfigurableActorCritic(context=runtime_context)
    restored.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=[{
            "type": "custom1",
            "in_features": "obs.shape",
            "out_features": "action.shape",
        }],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
        module_artifacts=artifacts,
    )

    assert restored.is_recurrent is True
    assert set(restored.get_recurrent_state(batch_size=2)) == {
        "custom1.memory"
    }
    torch.testing.assert_close(
        restored.actor.custom1.memory.gru.weight_ih_l0,
        torch.full_like(
            restored.actor.custom1.memory.gru.weight_ih_l0, 0.75
        ),
    )


@pytest.mark.parametrize(
    ("assertions", "message"),
    [
        ({"in_features": 4}, "configured inputs provide 3"),
        ({"out_features": 7}, "saved module produces 8"),
    ],
)
def test_artifact_module_validates_declared_interface(
    runtime_context: RuntimeContext,
    assertions: dict[str, int],
    message: str,
) -> None:
    module = torch.nn.Linear(3, 8)
    artifact = {
        "config": {
            "type": "linear",
            "in_features": 3,
            "out_features": 8,
        },
        "state_dict": module.state_dict(),
    }
    policy = ConfigurableActorCritic(context=runtime_context)

    with pytest.raises(ValueError, match=message):
        policy.config_update(
            component=make_component(), obs_dim=3, action_dim=2,
            actor=[
                {"type": "custom1", **assertions},
                {"type": "linear", "in_features": 8,
                 "out_features": "action.shape"},
            ],
            critic=CRITIC_CONFIG,
            distribution={"type": "diagonal_gaussian"},
            module_artifacts={"actor.custom1": artifact},
        )


def test_composite_module_cold_starts_from_policy_checkpoint(
    runtime_context: RuntimeContext,
) -> None:
    first = ConfigurableActorCritic(context=runtime_context)
    first.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=[{
            "name": "custom1",
            "type": "sequential",
            "modules": [
                {"type": "linear", "in_features": 3,
                 "out_features": 8},
                {"type": "elu"},
            ],
        }, {
            "type": "linear", "in_features": 8,
            "out_features": "action_dim",
        }],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    with torch.no_grad():
        first.actor.custom1[0].weight.fill_(0.625)
    checkpoint = first.checkpoint_state_dict()

    restored = ConfigurableActorCritic(context=runtime_context)
    restored.prepare_checkpoint_load(checkpoint)
    restored.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=[
            {"type": "custom1"},
            {"type": "linear", "in_features": 8,
             "out_features": "action_dim"},
        ],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    restored.load_checkpoint_state_dict(checkpoint)

    torch.testing.assert_close(
        restored.actor.custom1[0].weight,
        torch.full_like(restored.actor.custom1[0].weight, 0.625),
    )


def test_named_modules_are_saved_as_portable_files(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=[
            {"name": "encoder", "type": "linear",
             "in_features": 3, "out_features": 8},
            {"type": "linear", "in_features": 8,
             "out_features": "action_dim"},
        ],
        critic=[
            {"name": "value", "type": "linear",
             "in_features": 3, "out_features": 1},
        ],
        distribution={"type": "diagonal_gaussian"},
    )

    paths = policy.save_module_artifacts(tmp_path / "modules")

    assert paths == [
        tmp_path / "modules" / "actor" / "encoder.pt",
        tmp_path / "modules" / "critic" / "value.pt",
    ]
    actor_artifact = torch.load(paths[0], weights_only=False)
    assert actor_artifact["config"]["type"] == "linear"
    assert set(actor_artifact) == {"config", "state_dict"}


def test_nested_named_modules_are_saved_as_portable_files(
    runtime_context: RuntimeContext,
    tmp_path: Path,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=[{
            "name": "custom1",
            "type": "sequential",
            "modules": [
                {"name": "memory", "type": "gru",
                 "input_size": 3, "hidden_size": 8},
                {"type": "linear", "in_features": 8,
                 "out_features": "action_dim"},
            ],
        }],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )

    paths = policy.save_module_artifacts(tmp_path / "modules")

    assert tmp_path / "modules" / "actor" / "custom1.pt" in paths
    assert (
        tmp_path / "modules" / "actor" / "custom1" / "memory.pt"
        in paths
    )


def test_network_selects_observation_fields_and_concatenates_branches(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=5,
        action_dim=2,
        observation_slices={
            "field_a": slice(0, 3),
            "field_b": slice(3, 5),
        },
        actor=[
            {
                "name": "A",
                "inputs": "obs.field_a",
                "type": "linear",
                "in_features": "obs.field_a.shape",
                "out_features": 10,
            },
            {
                "name": "B",
                "inputs": "obs.field_b",
                "type": "linear",
                "in_features": "obs.field_b.shape",
                "out_features": 5,
            },
            {
                "name": "C",
                "inputs": ["A", "B"],
                "type": "linear",
                "in_features": 15,
                "out_features": "action_dim",
            },
        ],
        critic=[
            {"type": "linear", "in_features": "obs_dim",
             "out_features": 1},
        ],
        distribution={"type": "diagonal_gaussian"},
    )
    obs = torch.randn(4, 5)

    expected = policy.actor.C(torch.cat([
        policy.actor.A(obs[:, :3]),
        policy.actor.B(obs[:, 3:]),
    ], dim=-1))

    torch.testing.assert_close(policy.actor_forward(obs), expected)
    assert policy.actor.A.in_features == 3
    assert policy.actor.B.in_features == 2


def test_network_slices_named_module_outputs(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(),
        obs_dim=3,
        action_dim=2,
        actor=[
            {"name": "A", "type": "linear", "in_features": 3,
             "out_features": 10},
            {"name": "B", "inputs": "A[0:6]", "type": "linear",
             "in_features": 6, "out_features": 4},
            {"name": "C", "inputs": "A[6:10]", "type": "linear",
             "in_features": 4, "out_features": 3},
            {"name": "E", "inputs": ["B", "C"], "type": "linear",
             "in_features": 7, "out_features": "action_dim"},
        ],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    obs = torch.randn(4, 3)

    a_output = policy.actor.A(obs)
    expected = policy.actor.E(torch.cat([
        policy.actor.B(a_output[..., 0:6]),
        policy.actor.C(a_output[..., 6:10]),
    ], dim=-1))

    torch.testing.assert_close(policy.actor_forward(obs), expected)


def test_network_rejects_out_of_bounds_output_slice(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)

    with pytest.raises(ValueError, match="exceeds its 10 available features"):
        policy.config_update(
            component=make_component(),
            obs_dim=3,
            action_dim=2,
            actor=[
                {"name": "A", "type": "linear", "in_features": 3,
                 "out_features": 10},
                {"inputs": "A[5:11]", "type": "linear",
                 "in_features": 6, "out_features": "action_dim"},
            ],
            critic=CRITIC_CONFIG,
            distribution={"type": "diagonal_gaussian"},
        )


def test_network_resolves_previous_module_output_features(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=[
            {"name": "A", "type": "linear", "in_features": "obs.shape",
             "out_features": 10},
            {"name": "B", "type": "linear",
             "in_features": "A.out_features",
             "out_features": "action_dim"},
        ],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )

    assert policy.actor.A.in_features == 3
    assert policy.actor.B.in_features == 10


def test_network_rejects_unknown_observation_field(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    with pytest.raises(KeyError, match="Unknown observation field"):
        policy.config_update(
            component=make_component(), obs_dim=3, action_dim=2,
            observation_slices={"known": slice(0, 3)},
            actor=[{
                "inputs": "obs.missing",
                "type": "linear",
                "in_features": 3,
                "out_features": "action_dim",
            }],
            critic=CRITIC_CONFIG,
            distribution={"type": "diagonal_gaussian"},
        )


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


def test_recurrent_sequence_supports_sliced_module_output(
    runtime_context: RuntimeContext,
) -> None:
    policy = ConfigurableActorCritic(context=runtime_context)
    policy.config_update(
        component=make_component(), obs_dim=3, action_dim=2,
        actor=[
            {"name": "memory", "type": "gru",
             "input_size": 3, "hidden_size": 8},
            {"inputs": "memory[0:4]", "type": "linear",
             "in_features": 4, "out_features": "action_dim"},
        ],
        critic=CRITIC_CONFIG,
        distribution={"type": "diagonal_gaussian"},
    )
    obs = torch.randn(3, 2, 3)
    actions = torch.zeros(3, 2, 2)

    evaluation, final_state = policy.evaluate_recurrent_sequences(
        obs=obs,
        actions=actions,
        initial_state=policy.get_recurrent_state(batch_size=2),
        reset_mask=torch.zeros(3, 2, dtype=torch.bool),
    )

    assert evaluation.log_prob.shape == (3, 2)
    assert final_state["memory"].shape == (2, 8)


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
