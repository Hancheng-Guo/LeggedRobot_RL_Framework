"""Playback command terms preserve the manager's normal update path."""

import torch
import pytest

from envs.tasks.managers.command import CommandManager


class Source:
    def __init__(self) -> None:
        self.current = {"lin_vel_x": 0.5, "lin_vel_y": -0.25, "ang_vel_z": 0.75}
        self.polls = 0

    def poll(self) -> None:
        self.polls += 1

    def values(self) -> dict[str, float]:
        return self.current


def test_playback_terms_use_constraints_and_restore_original_sampling(
    runtime_context, model_context,
) -> None:
    manager = CommandManager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        terms={
            name: {"type": "UniformOnReset", "params": {"min_value": 0.0, "max_value": 0.0}}
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        },
        constraints={"lin_vel_x": {"operator": "<=", "expression": "0.3"}},
    )
    source = Source()
    manager.activate_playback_commands(source, env_index=1)
    manager.reset()

    torch.testing.assert_close(manager.command["lin_vel_x"][:, 0], torch.tensor([0.0, 0.3, 0.0]))
    torch.testing.assert_close(manager.command["lin_vel_y"][:, 0], torch.tensor([0.0, -0.25, 0.0]))
    assert source.polls == 0

    source.current["lin_vel_x"] = -0.4
    manager.update(None)  # type: ignore[arg-type]  # UniformOnReset ignores context.
    torch.testing.assert_close(manager.last_command["lin_vel_x"][:, 0], torch.tensor([0.0, 0.3, 0.0]))
    assert manager.command["lin_vel_x"][1, 0] == -0.4
    assert source.polls == 0

    manager.deactivate_playback_commands()
    manager.reset(torch.tensor([1]))
    assert manager.command["lin_vel_x"][1, 0] == 0.0


def test_partial_reset_keeps_playback_command_when_another_env_resets(
    runtime_context, model_context,
) -> None:
    manager = CommandManager(
        num_envs=3,
        context=runtime_context,
        model_context=model_context,
        terms={
            name: {"type": "UniformOnReset", "params": {"min_value": 0.0, "max_value": 0.0}}
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        },
    )
    manager.activate_playback_commands(Source(), env_index=1)
    manager.reset()
    manager.reset(torch.tensor([2]))
    assert manager.command["lin_vel_x"][1, 0] == 0.5


def test_deactivate_before_first_reset_does_not_resample_commands(
    runtime_context, model_context, monkeypatch,
) -> None:
    manager = CommandManager(
        num_envs=1,
        context=runtime_context,
        model_context=model_context,
        terms={
            "lin_vel_x": {
                "type": "UniformOnReset",
                "params": {"min_value": 0.0, "max_value": 0.0},
            },
        },
    )
    original = manager.terms["lin_vel_x"]
    source = Source()
    source.current = {"lin_vel_x": 0.5}
    monkeypatch.setattr(original, "reset", lambda env_ids: pytest.fail("unexpected reset"))

    manager.activate_playback_commands(source, env_index=0)
    manager.deactivate_playback_commands()

    assert manager.terms["lin_vel_x"] is original


def test_playback_wraps_all_configured_commands(runtime_context, model_context) -> None:
    manager = CommandManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "body_height": {
                "type": "UniformOnReset",
                "params": {"min_value": 0.0, "max_value": 0.0},
            },
        },
    )
    original = manager.terms["body_height"]
    source = Source()
    source.current = {"body_height": 0.2}

    manager.activate_playback_commands(source, env_index=1)
    manager.reset()
    torch.testing.assert_close(manager.command["body_height"][:, 0], torch.tensor([0.0, 0.2]))
    assert source.polls == 0

    manager.deactivate_playback_commands()
    assert manager.terms["body_height"] is original
    assert manager.command["body_height"][1, 0] == 0.0


def test_playback_rejects_incomplete_source(runtime_context, model_context) -> None:
    manager = CommandManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            name: {"type": "UniformOnReset", "params": {"min_value": 0.0, "max_value": 0.0}}
            for name in ("lin_vel_x", "body_height")
        },
    )
    original_terms = dict(manager.terms)
    source = Source()
    source.current = {"lin_vel_x": 0.5}

    with pytest.raises(ValueError, match="body_height"):
        manager.activate_playback_commands(source, env_index=1)

    assert manager.terms == original_terms
    assert manager._playback_commands is None


def test_playback_uses_fixed_vector_commands(runtime_context, model_context) -> None:
    manager = CommandManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "lin_vel_x": {"type": "UniformOnReset", "params": {"min_value": 0.0, "max_value": 0.0}},
            "foot_phase_real": {"type": "UniformOnReset", "params": {"dim": 4, "min_value": 0.0, "max_value": 0.0}},
        },
    )
    source = Source()
    source.current = {"lin_vel_x": 0.5}
    original_terms = dict(manager.terms)
    manager.activate_playback_commands(
        source, env_index=1,
        fixed_commands={"foot_phase_real": (1.0, 1.0, 1.0, 1.0)},
    )
    manager.reset()
    torch.testing.assert_close(manager.command["lin_vel_x"][:, 0], torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(
        manager.command["foot_phase_real"],
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    manager.deactivate_playback_commands()
    assert manager.terms == original_terms
    torch.testing.assert_close(manager.command["foot_phase_real"], torch.zeros(2, 4))


@pytest.mark.parametrize(
    ("fixed_commands", "message"),
    [
        ({"lin_vel_x": (0.0,)}, "two sources"),
        ({"foot_phase_real": (1.0,)}, "dimension mismatch"),
    ],
)
def test_playback_rejects_conflicting_or_wrong_size_fixed_commands(
    runtime_context, model_context, fixed_commands, message,
) -> None:
    manager = CommandManager(
        num_envs=1,
        context=runtime_context,
        model_context=model_context,
        terms={
            "lin_vel_x": {"type": "UniformOnReset", "params": {"min_value": 0.0, "max_value": 0.0}},
            "foot_phase_real": {"type": "UniformOnReset", "params": {"dim": 4, "min_value": 0.0, "max_value": 0.0}},
        },
    )
    source = Source()
    source.current = {"lin_vel_x": 0.5}
    with pytest.raises(ValueError, match=message):
        manager.activate_playback_commands(source, env_index=0, fixed_commands=fixed_commands)
