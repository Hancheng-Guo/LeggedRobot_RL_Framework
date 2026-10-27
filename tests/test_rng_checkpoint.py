import random
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from envs import BaseEnv
from rl.algorithms import OnPolicyAlgorithm
from runners import OnPolicyRunner, RestoreMode
from utils import Component


def draw_random_values():
    return random.random(), np.random.random(3), torch.rand(3)


def assert_same_draw(expected, actual):
    assert expected[0] == actual[0]
    np.testing.assert_array_equal(expected[1], actual[1])
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)


@pytest.fixture(autouse=True)
def preserve_global_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


@pytest.fixture
def make_rng_runner(runtime_context):
    def make():
        runner = OnPolicyRunner(runtime_context)
        runner.current_iteration = 0
        runner.max_iterations = 1
        runner.rollout_length = 1
        reset_draws = []
        obs = torch.zeros(1, 1)

        def reset():
            reset_draws.append(draw_random_values())
            return obs

        def step(action):
            draw_random_values()
            return obs, obs, torch.ones(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}

        # A generic environment: RNG persistence must not depend on curriculum.
        runner.environment = cast(BaseEnv, SimpleNamespace(
            num_envs=1, reset=reset, step=step, close=lambda: None,
            checkpoint_state_dict=lambda: {}, load_checkpoint_state_dict=lambda state: None,
        ))
        runner.algorithm = cast(OnPolicyAlgorithm, SimpleNamespace(
            checkpoint_state_dict=lambda: {},
            load_checkpoint_state_dict=lambda state, load_optimizer: draw_random_values(),
            save_module_artifacts=lambda directory: [], close=lambda: None,
            set_train_mode=lambda: None, set_eval_mode=lambda: None,
            act=lambda obs, deterministic: SimpleNamespace(action=torch.rand(1, 1)),
            reset_policy_state=lambda env_ids: None,
        ))
        return runner, reset_draws
    return make


def test_ordinary_resume_restores_rng_before_environment_reset(make_rng_runner, tmp_path):
    source, _ = make_rng_runner()
    path = source.save(tmp_path / "rng.pt")
    expected_reset = draw_random_values()
    expected_after_reset = draw_random_values()
    resumed, draws = make_rng_runner()
    resumed.prepare_checkpoint_load(path)
    resumed.load()  # Normal resume uses the default mode; no additional option.
    draw_random_values()  # Initialization between load and train must not shift resume RNG.
    resumed.train()
    assert_same_draw(expected_reset, draws[0])
    assert_same_draw(expected_after_reset, draw_random_values())
    assert resumed._pending_rng_state is None
    # A later train() call must not rewind to the checkpoint again.
    next_state = resumed._rng_state_dict()
    expected_next_reset = draw_random_values()
    resumed._restore_rng_state(next_state)
    resumed.train()
    assert_same_draw(expected_next_reset, draws[1])


@pytest.mark.parametrize("mode", [RestoreMode.EVALUATE, RestoreMode.ADVANCE_STAGE])
def test_other_restore_modes_do_not_rewind_rng(make_rng_runner, tmp_path, mode):
    source, _ = make_rng_runner()
    path = source.save(tmp_path / "rng.pt")
    resumed, draws = make_rng_runner()
    resumed.prepare_checkpoint_load(path)
    resumed.load(mode=mode)
    assert resumed._pending_rng_state is None
    state = resumed._rng_state_dict()
    expected_reset = draw_random_values()
    resumed._restore_rng_state(state)
    resumed.train()
    assert_same_draw(expected_reset, draws[0])


def test_resaving_pending_resume_keeps_original_rng(make_rng_runner, tmp_path):
    source, _ = make_rng_runner()
    path = source.save(tmp_path / "original.pt")
    expected_reset = draw_random_values()
    resumed, _ = make_rng_runner()
    resumed.prepare_checkpoint_load(path)
    resumed.load()
    draw_random_values()
    resaved = resumed.save(tmp_path / "resaved.pt")
    final, draws = make_rng_runner()
    final.prepare_checkpoint_load(resaved)
    final.load()
    final.train()
    assert_same_draw(expected_reset, draws[0])


def test_evaluation_preserves_training_rng_for_save_and_next_train(make_rng_runner, tmp_path):
    runner, draws = make_rng_runner()
    initial_state = runner._rng_state_dict()
    expected_reset = draw_random_values()
    expected_after_reset = draw_random_values()
    runner._restore_rng_state(initial_state)
    runner.test(num_episodes=1)
    runner.test(num_episodes=1)
    path = runner.save(tmp_path / "after_evaluation.pt")
    resumed, resumed_draws = make_rng_runner()
    resumed.prepare_checkpoint_load(path)
    resumed.load()
    resumed.train()
    assert_same_draw(expected_reset, resumed_draws[0])
    runner.train()
    assert_same_draw(expected_reset, draws[-1])
    assert_same_draw(expected_after_reset, draw_random_values())


def test_checkpoint_without_rng_leaves_current_sequence_unchanged(make_rng_runner, tmp_path):
    source, _ = make_rng_runner()
    state = source.checkpoint_state_dict()
    state.pop("rng")
    path = tmp_path / "without_rng.pt"
    torch.save({"runner": state}, path)
    resumed, draws = make_rng_runner()
    resumed.prepare_checkpoint_load(path)
    resumed.load()
    current = resumed._rng_state_dict()
    expected_reset = draw_random_values()
    resumed._restore_rng_state(current)
    resumed.train()
    assert_same_draw(expected_reset, draws[0])


def test_rng_snapshot_is_independent_and_capture_does_not_advance_rng(make_rng_runner):
    runner, _ = make_rng_runner()
    state = runner._rng_state_dict()
    expected = draw_random_values()
    runner._restore_rng_state(state)
    independent = runner._rng_state_dict()
    state["numpy"][1].fill(0)
    state["torch_cpu"].zero_()
    assert_same_draw(expected, draw_random_values())
    runner._restore_rng_state(independent)
    assert_same_draw(expected, draw_random_values())


def test_cpu_checkpoint_does_not_initialize_cuda(make_rng_runner, monkeypatch):
    runner, _ = make_rng_runner()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def unexpected_cuda_read():
        pytest.fail("CPU checkpoint must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", unexpected_cuda_read)
    assert runner._rng_state_dict()["torch_cuda"] == []


@pytest.mark.parametrize("device_count", [0, 1, 2, 3])
def test_cuda_rng_restores_available_device_indices(make_rng_runner, monkeypatch, device_count):
    runner, _ = make_rng_runner()
    saved_cuda = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: saved_cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    restored = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: restored.append((device, state.clone())))
    state = runner._rng_state_dict()
    saved_cuda[0].zero_()
    runner._restore_rng_state(state)
    assert len(restored) == min(device_count, 2)
    for index, value in restored:
        assert value.device.type == "cpu"
        torch.testing.assert_close(value, state["torch_cuda"][index])
    assert state["torch_cuda"][0].tolist() == [1, 2]


def test_stage_change_and_new_load_clear_pending_rng(make_rng_runner, monkeypatch, tmp_path):
    runner, _ = make_rng_runner()
    runner._pending_rng_state = runner._rng_state_dict()
    runner._checkpoint_rng_snapshot = runner._rng_state_dict()
    monkeypatch.setattr(runner, "_build_environment", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_build_algorithm", lambda **kwargs: None)
    runner.config_update(Component(None, None, None, None, None, None), stage_index=1)
    assert runner._pending_rng_state is None
    assert runner._checkpoint_rng_snapshot is None
    path = runner.save(tmp_path / "new.pt")
    runner._pending_rng_state = runner._rng_state_dict()
    runner._checkpoint_rng_snapshot = runner._rng_state_dict()
    runner.prepare_checkpoint_load(path)
    assert runner._pending_rng_state is None
    assert runner._checkpoint_rng_snapshot is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_tensor_rng_states_can_load_from_cuda_mapping(make_rng_runner, tmp_path):
    runner, _ = make_rng_runner()
    torch.rand(3, device="cuda")
    path = runner.save(tmp_path / "cuda_rng.pt")
    expected_cpu = torch.rand(3)
    expected_cuda = torch.rand(3, device="cuda")
    state = torch.load(path, map_location="cuda", weights_only=False)["runner"]["rng"]
    runner._restore_rng_state(state)
    torch.testing.assert_close(torch.rand(3), expected_cpu, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(3, device="cuda"), expected_cuda, rtol=0, atol=0)
