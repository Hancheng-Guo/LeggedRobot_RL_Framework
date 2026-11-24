from copy import deepcopy

import pytest
import torch

from envs.simulators.state import SimulatorState


def make_state() -> SimulatorState:
    values = {
        "qpos": torch.arange(6).reshape(2, 3),
        "qvel": torch.arange(4).reshape(2, 2),
        "qacc": torch.arange(4).reshape(2, 2),
        "ctrl": torch.arange(4).reshape(2, 2),
        "geom_xpos": torch.arange(12).reshape(2, 2, 3),
        "geom_xvel": torch.arange(24).reshape(2, 2, 6),
        "actuator_force": torch.arange(4).reshape(2, 2),
        "base_lin_vel_body": torch.arange(6).reshape(2, 3),
        "base_ang_vel_body": torch.arange(6).reshape(2, 3),
        "contact_geom_ids": torch.arange(4).reshape(2, 1, 2),
        "contact_forces": torch.arange(12).reshape(2, 1, 6),
        "foot_ground_contact": torch.tensor([[True], [False]]),
        "foot_contact_normal_force": torch.tensor([[12.0], [0.0]]),
    }
    return SimulatorState(**values)


def test_simulator_state_supports_attributes_and_dictionary_export():
    state = make_state()

    assert state.as_dict()["qpos"] is state.qpos
    assert state.as_dict()["geom_xvel"] is state.geom_xvel


def test_simulator_state_exposes_term_outputs_as_attributes():
    state = make_state()
    output = torch.tensor([[1.0], [2.0]])
    state.term_values = {
        "scan_height": output, "qpos": output, "base_ground_height": output,
    }
    assert state.scan_height is output
    assert state.qpos is not output
    assert state.base_ground_height is None
    exported = state.as_dict()
    assert exported["scan_height"] is output
    assert exported["qpos"] is state.qpos
    assert "base_ground_height" not in exported
    torch.testing.assert_close(state.select(torch.tensor([1])).scan_height, output[1:])
    torch.testing.assert_close(deepcopy(state).scan_height, output)


def test_simulator_state_reports_missing_term_outputs():
    state = make_state()
    assert not hasattr(state, "scan_height")
    with pytest.raises(AttributeError, match="scan_height"):
        _ = state.scan_height


def test_simulator_state_preserves_explicit_none_term_outputs():
    state = make_state()
    state.term_values["scan_height"] = None

    for result in (state, state.clone(), state.select(torch.tensor([1]))):
        assert hasattr(result, "scan_height")
        assert result.scan_height is None
        assert "scan_height" not in result.as_dict()


def test_simulator_state_clone_owns_independent_storage():
    state = make_state()
    snapshot = state.clone()

    state.qpos.zero_()

    assert torch.count_nonzero(snapshot.qpos) > 0
    assert snapshot.qpos.data_ptr() != state.qpos.data_ptr()


def test_simulator_state_selects_environments():
    state = make_state()
    selected = state.select(torch.tensor([1]))

    for name, value in selected.as_dict().items():
        torch.testing.assert_close(value, state.as_dict()[name][1:2])
