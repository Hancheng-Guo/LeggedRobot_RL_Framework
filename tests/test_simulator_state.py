import torch

from envs.simulators.utils.state import SimulatorState


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
