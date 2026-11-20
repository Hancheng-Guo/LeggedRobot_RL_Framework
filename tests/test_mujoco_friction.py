"""MuJoCo contact, isolation and recovery for independent friction randomization."""
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
import yaml

from envs.simulators.mujoco import MujocoSimulator
from envs.tasks.locomotion import LocomotionTaskLogic
from envs.tasks.managers.curriculum import CurriculumSampler
from envs.tasks.managers.randomization import RandomizationManager
from envs.tasks.managers.randomization.terms import FrictionRandomizationTerm
from envs.vector_env import VectorEnv
from utils import Component, ComponentInfo, load_yaml

pytestmark = pytest.mark.mujoco
ROOT = Path(__file__).resolve().parents[1]


def friction(simulator, env_id):
    return simulator.models[env_id].geom_friction.copy()


@pytest.mark.parametrize(("contact_config", "message"), [
    ('<option><flag override="enable"/></option>', "contact override replaces geom friction"),
    ('<contact><pair geom1="floor" geom2="foot"/></contact>', "explicit contact pairs"),
])
def test_friction_rejects_contact_overrides(runtime_context, tmp_path, contact_config, message):
    path = tmp_path / "slider.xml"
    path.write_text(f"""
<mujoco>
  {contact_config}
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body pos="0 0 .099">
      <freejoint/>
      <geom name="foot" type="box" size=".15 .15 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""", encoding="utf-8")
    sim = MujocoSimulator(runtime_context)
    try:
        sim.config_update(Component(None, None, None, None, None, None), num_envs=1,
                          model_path=path, foot_geom_names=("foot",), floor_geom_names=("floor",),
                          sim_dt=.002, frame_skip=1)
        manager = RandomizationManager(1, runtime_context, terms={
            "friction": {"type": "Fixed", "params": {"factors": [2.]}},
        })
        with pytest.raises(ValueError, match=message):
            manager.bind(sim)
    finally:
        sim.close()


@pytest.mark.parametrize("floor_priority", [0, 1, 2])
def test_floor_and_foot_friction_and_actual_sliding(runtime_context, tmp_path, floor_priority):
    path = tmp_path / "slider.xml"
    path.write_text(f"""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1" friction=".25 .03 .004" priority="{floor_priority}" condim="3"/>
    <geom name="other" type="sphere" pos="5 5 1" size=".1" friction=".4 .02 .003"/>
    <body name="slider" pos="0 0 .099">
      <freejoint/>
      <geom name="foot" type="box" size=".15 .15 .1" mass="1"
            friction=".2 .02 .003" priority="1" condim="3"/>
    </body>
  </worldbody>
</mujoco>
""", encoding="utf-8")
    sim = MujocoSimulator(runtime_context)
    try:
        sim.config_update(Component(None, None, None, None, None, None), num_envs=2,
                          model_path=path, foot_geom_names=("foot",), floor_geom_names=("floor",),
                          sim_dt=.002, frame_skip=1)
        original = friction(sim, 0)
        manager = RandomizationManager(2, runtime_context, terms={
            "friction": {"type": "Fixed", "params": {"factors": [3.]}},
        })
        manager.bind(sim)
        expected = original.copy()
        geom_ids = sim.model_context.floor_geom_ids.tolist() + sim.model_context.foot_geom_ids.tolist()
        expected[geom_ids, 0] *= 3.
        for _ in range(2):
            manager.bind(sim)
            manager.before_simulator_reset(torch.tensor([1]))
            sim.reset()
            manager.commit_simulator_reset()
            np.testing.assert_array_equal(friction(sim, 0), original)
            np.testing.assert_array_equal(friction(sim, 1), expected)
        for model, data in zip(sim.models, sim.datas):
            data.qvel[0] = 2.
            mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        baseline_contact = .25 if floor_priority >= 1 else .2
        assert sim.datas[0].contact[0].friction[0] == pytest.approx(baseline_contact)
        assert sim.datas[1].contact[0].friction[0] == pytest.approx(3. * baseline_contact)
        for _ in range(65):
            for model, data in zip(sim.models, sim.datas):
                mujoco.mj_step(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        assert sim.datas[1].qvel[0] < sim.datas[0].qvel[0] - .3
    finally:
        sim.close()


def make_environment(context, task_path):
    env = VectorEnv(replace(context, load_dir=ROOT, save_dir=ROOT))
    component = Component(None, None, None, None,
        ComponentInfo("mujoco", ROOT / "configs/simulators/mujoco_anymal_c_rgb_array.yaml"),
        ComponentInfo("locomotion", task_path))
    try:
        env.config_update(component, num_envs=3, max_episode_steps=25)
    except Exception:
        env.close()
        raise
    return env


def term(env):
    task = cast(LocomotionTaskLogic, env.task)
    return cast(FrictionRandomizationTerm, task.randomization_manager.terms["friction"])


@pytest.mark.parametrize("partial_reset", [False, True])
def test_friction_only_rollback_restores_contact_data(
    runtime_context, tmp_path, monkeypatch, partial_reset,
):
    config = load_yaml(ROOT / "configs/tasks/locomotion_anymal_c_phase.yaml")
    config["randomization_manager_config"]["terms"] = {
        "friction": {"type": "Uniform", "params": {"factors": [.5, 3.]}},
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = make_environment(runtime_context, path)
    try:
        sim = cast(MujocoSimulator, env.simulator)
        for model in sim.models:
            model.key_qpos[sim.reset_keyframe_id, 2] -= .04
        friction_term = term(env)
        monkeypatch.setattr(friction_term, "_sample", lambda env_ids: [3.] * len(env_ids))
        obs = env.reset()
        for model, data in zip(sim.models, sim.datas):
            data.qvel[0] = .1
            mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
            assert data.ncon > 0
        before = friction_term.checkpoint_state_dict()
        originals = [friction(sim, i) for i in range(env.num_envs)]
        contacts = [data.contact.friction.copy() for data in sim.datas]
        velocities = [data.qvel.copy() for data in sim.datas]

        def fail_observation(*args, **kwargs):
            assert not np.array_equal(sim.datas[1].contact.friction, contacts[1])
            raise RuntimeError("injected observation failure")

        with monkeypatch.context() as injected:
            injected.setattr(friction_term, "_sample", lambda env_ids: [.5] * len(env_ids))
            injected.setattr(env.task, "compute_observation", fail_observation)
            with pytest.raises(RuntimeError, match="injected observation failure"):
                if partial_reset:
                    env._reset_done_envs(
                        obs, torch.tensor([False, True, False]), torch.zeros(3, dtype=torch.bool),
                    )
                else:
                    env.reset()
        assert friction_term.checkpoint_state_dict() == before
        for i, data in enumerate(sim.datas):
            np.testing.assert_array_equal(friction(sim, i), originals[i])
            np.testing.assert_array_equal(data.contact.friction, contacts[i])
            np.testing.assert_array_equal(data.qvel, velocities[i])
        assert torch.isfinite(env.reset()).all()
    finally:
        env.close()


@pytest.mark.parametrize("strategy", ["Uniform", "Curriculum"])
def test_checkpoint_partial_reset_and_fixed_playback(runtime_context, tmp_path, strategy):
    path = tmp_path / "task.yaml"
    config = load_yaml(ROOT / "configs/tasks/locomotion_anymal_c_phase.yaml")
    factors = [.2, .8, 1.1]
    params: dict[str, Any] = {"factors": factors}
    if strategy == "Curriculum":
        params["curriculum"] = "locomotion"
    config["randomization_manager_config"]["terms"]["friction"] = {
        "type": strategy, "params": params, "seed": 17,
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = make_environment(runtime_context, path)
    restored = make_environment(runtime_context, path)
    playback = None
    try:
        sim = cast(MujocoSimulator, env.simulator)
        baselines = [friction(sim, i) for i in range(3)]
        obs = env.reset()
        floor_ids = sim.model_context.floor_geom_ids.tolist()
        assert len(floor_ids) == 5  # heightfield plus four peripheral slabs
        if strategy == "Curriculum":
            task = cast(LocomotionTaskLogic, env.task)
            sampler = cast(CurriculumSampler, task.curriculum_manager.get_term("locomotion"))
            lower, _ = sampler.get_bounds("friction")
            assert term(env).factors == [factors[int(i)] for i in lower.flatten()]
        geom_ids = floor_ids + sim.model_context.foot_geom_ids.tolist()
        for i, factor in enumerate(term(env).factors):
            expected = baselines[i].copy()
            expected[geom_ids, 0] *= factor
            np.testing.assert_array_equal(friction(sim, i), expected)
        saved = env.checkpoint_state_dict()
        restored.load_checkpoint_state_dict(saved)
        restored.reset()
        assert term(restored).checkpoint_state_dict() == term(env).checkpoint_state_dict()
        for i in range(3):
            np.testing.assert_array_equal(friction(cast(MujocoSimulator, restored.simulator), i), friction(sim, i))

        env.set_training_mode(False)
        before = term(env).checkpoint_state_dict()
        env.reset()
        assert term(env).checkpoint_state_dict() == before
        playback = cast(VectorEnv, env._create_playback_environment())
        playback.reset()
        assert term(playback).factors == [term(env).factors[env.playback_env_index]]
        np.testing.assert_array_equal(friction(cast(MujocoSimulator, playback.simulator), 0),
                                      friction(sim, env.playback_env_index))
        env.set_training_mode(True)
        others = [friction(sim, i) for i in (0, 2)]
        env._reset_done_envs(obs, torch.tensor([False, True, False]), torch.zeros(3, dtype=torch.bool))
        for i, value in zip((0, 2), others):
            np.testing.assert_array_equal(friction(sim, i), value)
        if strategy == "Uniform":
            # Replay the same partial reset after the restoration reset.
            restored._reset_done_envs(obs, torch.tensor([False, True, False]), torch.zeros(3, dtype=torch.bool))
            assert term(restored).checkpoint_state_dict() == term(env).checkpoint_state_dict()
    finally:
        if playback is not None:
            playback.close()
        restored.close()
        env.close()


@pytest.mark.parametrize("friction_first", [False, True])
@pytest.mark.parametrize("failure", ["write", "observation"])
def test_friction_rollback_with_terrain(runtime_context, tmp_path, monkeypatch, friction_first, failure):
    config = load_yaml(ROOT / "configs/tasks/locomotion_anymal_c_phase.yaml")
    terms = config["randomization_manager_config"]["terms"]
    friction_config = {"type": "Uniform", "params": {"factors": [.2, .8]}}
    config["randomization_manager_config"]["terms"] = (
        {"friction": friction_config, **terms} if friction_first else {**terms, "friction": friction_config}
    )
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = make_environment(runtime_context, path)
    try:
        env.reset()
        sim = cast(MujocoSimulator, env.simulator)
        original = [friction(sim, i) for i in range(3)]
        terrain = [model.hfield_data.copy() for model in sim.models]
        before = term(env).checkpoint_state_dict()
        apply = sim.apply_friction

        with monkeypatch.context() as injected:
            if failure == "write":
                def fail_write(env_ids, factors):
                    apply(env_ids[:1], [2.])
                    raise RuntimeError("injected friction write failure")
                injected.setattr(sim, "apply_friction", fail_write)
            else:
                def fail_observation(*args, **kwargs):
                    raise RuntimeError("injected observation failure")
                injected.setattr(env.task, "compute_observation", fail_observation)
            with pytest.raises(RuntimeError, match=f"injected .*{failure} failure"):
                env.reset()
        assert term(env).checkpoint_state_dict() == before
        for i, model in enumerate(sim.models):
            np.testing.assert_array_equal(friction(sim, i), original[i])
            np.testing.assert_array_equal(model.hfield_data, terrain[i])
        assert torch.isfinite(env.reset()).all()
    finally:
        env.close()
