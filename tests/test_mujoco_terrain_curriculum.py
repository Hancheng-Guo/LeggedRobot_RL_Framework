"""Real physics rollback and checkpoint reconstruction for phase-four consumers."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from envs.vector_env import VectorEnv
from envs.tasks.locomotion import LocomotionTaskLogic
from envs.simulators.mujoco import MujocoSimulator
from envs.tasks.managers.randomization.terms.terrain import TerrainRandomizationTerm
from utils import Component, ComponentInfo
from test_terrain_curriculum import assert_equal, curriculum_runtime_state

pytestmark = pytest.mark.mujoco
ROOT = Path(__file__).resolve().parents[1]


def terrain(env):
    return cast(TerrainRandomizationTerm, env.task.randomization_manager.terms['terrain'])


def environment(context):
    env = VectorEnv(replace(context, load_dir=ROOT, save_dir=ROOT))
    component = Component(None, None, None, None,
        ComponentInfo('mujoco', ROOT / 'configs/simulators/mujoco_anymal_c_rgb_array.yaml'),
        ComponentInfo('locomotion', ROOT / 'configs/tasks/locomotion_anymal_c_phase.yaml'))
    env.config_update(component, num_envs=3, max_episode_steps=50)
    env.reset()
    return env


def test_observation_failure_restores_real_geometry_physics_and_course(runtime_context, monkeypatch):
    env = environment(runtime_context)
    try:
        task = cast(LocomotionTaskLogic, env.task)
        term = terrain(env)
        term.request_reallocation(torch.tensor([1]))
        state = deepcopy(env.checkpoint_state_dict())
        buffers = curriculum_runtime_state(task.curriculum_manager)
        models = [m.hfield_data.copy() for m in cast(MujocoSimulator, env.simulator).models]
        positions = env.simulator.get_state().qpos.clone()
        velocities = env.simulator.get_state().qvel.clone()
        original = env.task.compute_observation
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('observation reset failed')
        monkeypatch.setattr(env.task, 'compute_observation', fail)
        with pytest.raises(RuntimeError, match='observation reset failed'):
            env.reset()
        assert_equal(state, env.checkpoint_state_dict())
        assert_equal(buffers, curriculum_runtime_state(task.curriculum_manager))
        torch.testing.assert_close(positions, env.simulator.get_state().qpos, rtol=0, atol=0)
        torch.testing.assert_close(velocities, env.simulator.get_state().qvel, rtol=0, atol=0)
        for model, heights in zip(cast(MujocoSimulator, env.simulator).models, models):
            np.testing.assert_array_equal(model.hfield_data, heights)
        monkeypatch.setattr(env.task, 'compute_observation', original)
        assert torch.isfinite(env.reset()).all()
    finally:
        env.close()


def test_actual_playback_and_restoration_keep_patch_and_height_buffer(runtime_context):
    env = environment(runtime_context)
    restored = environment(runtime_context)
    playback = None
    try:
        before = env.checkpoint_state_dict()
        restored.load_checkpoint_state_dict(before)
        restored.reset()
        assert_equal(before, restored.checkpoint_state_dict())
        for first, second in zip(cast(MujocoSimulator, env.simulator).models, cast(MujocoSimulator, restored.simulator).models):
            np.testing.assert_array_equal(first.hfield_data, second.hfield_data)
        playback = cast(VectorEnv, env._create_playback_environment())
        playback.reset()
        expected = terrain(env).patches[env.playback_env_index]
        for _ in range(3):
            assert terrain(playback).patches[0] == expected
            playback.reset()
        assert_equal(before, env.checkpoint_state_dict())
    finally:
        if playback is not None:
            playback.close()
        restored.close()
        env.close()


def test_environment_supplies_ascending_partial_reset_ids(runtime_context, monkeypatch):
    env = environment(runtime_context)
    try:
        observation = env.reset()
        selected = []
        original = env.task.before_simulator_reset
        def record(env_ids=None):
            assert env_ids is not None
            selected.append(env_ids.tolist())
            return original(env_ids)
        monkeypatch.setattr(env.task, 'before_simulator_reset', record)
        result = env._reset_done_envs(observation, torch.tensor([True, False, True]),
                                     torch.zeros(3, dtype=torch.bool))
        assert selected == [[0, 2]]
        assert torch.isfinite(result).all()
    finally:
        env.close()


def test_every_configured_term_has_its_manager_key_as_name(runtime_context):
    env = environment(runtime_context)
    try:
        task = cast(LocomotionTaskLogic, env.task)
        for manager_name in (*task.CHECKPOINT_MANAGERS, 'curriculum_manager', 'randomization_manager'):
            manager = getattr(task, manager_name)
            for name, term in manager.terms.items():
                assert term.name == name
    finally:
        env.close()
