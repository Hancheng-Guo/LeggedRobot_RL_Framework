"""Stage-3 real terrain/task integration, including safe out-of-bounds auto-reset."""
from dataclasses import replace
from pathlib import Path
from typing import cast
import numpy as np
import pytest
import torch
import yaml
from envs.vector_env import VectorEnv
from envs.tasks.managers.randomization.terms.terrain import TerrainRandomizationTerm
from envs.simulators.terrain import TerrainPatch
from envs.tasks.managers.reward.terms.contact import IllegalContactL1
from envs.tasks.managers.termination.terms.contact import BodyContact
from utils import Component, ComponentInfo, load_yaml

pytestmark = pytest.mark.mujoco
ROOT = Path(__file__).resolve().parents[1]


def terrain_task_config():
    config = load_yaml(ROOT/'configs/tasks/locomotion_unitree_go1_standing.yaml')
    config['randomization_manager_config'] = load_yaml(
        ROOT/'configs/tasks/locomotion_anymal_c_phase.yaml')['randomization_manager_config']
    return config


def terrain_environment(runtime_context, task_path, num_envs=3):
    env = VectorEnv(replace(runtime_context, load_dir=ROOT, save_dir=ROOT))
    component = Component(None,None,None,None,
        ComponentInfo('mujoco', ROOT/'configs/simulators/mujoco_unitree_go1_rgb_array.yaml'),
        ComponentInfo('locomotion', task_path))
    env.config_update(component, num_envs=num_envs, max_episode_steps=50)
    env.reset()
    return env


@pytest.fixture
def env(runtime_context, tmp_path):
    config = terrain_task_config()
    config['termination_manager_config']['terms']['terrain_bounds'] = {}
    path = tmp_path/'task.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    instance = terrain_environment(runtime_context, path)
    yield instance
    instance.close()


def task_context(env, env_ids=None):
    return env.task.build_task_context(env.simulator.get_state(env_ids),
        env.current_episode_steps if env_ids is None else env.current_episode_steps[env_ids],
        env.simulator.control_dt, env_ids)


def test_equivalent_stances_on_actual_slope_have_equal_height_signals(runtime_context, tmp_path):
    config = terrain_task_config()
    config['randomization_manager_config']['terms']['terrain']['types'] = {'slope':{'max_slope':.3}}
    path = tmp_path/'task.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    env = terrain_environment(runtime_context, path, 2)
    try:
        term = cast(TerrainRandomizationTerm, env.task.randomization_manager.terms['terrain'])
        assert term.patches[0] is not None
        geometry = term.patches[0].geometry
        slope = replace(geometry, heights=tuple(tuple(.2*x for x in geometry.x) for y in geometry.y))
        env.simulator.apply_terrain(torch.arange(2), [slope, slope])
        positions = torch.tensor([[-1.2,0.,.08],[1.2,0.,.56]],dtype=torch.float64)
        env.simulator.set_root_positions(torch.arange(2), positions)
        context = task_context(env)
        assert context.state.ground_valid is not None
        assert context.state.ground_valid.all()
        torch.testing.assert_close(context.state.base_ground_height, torch.tensor([-.24,.24]), atol=1e-6, rtol=0)
        obs = env.task.observation_manager.terms
        torch.testing.assert_close(obs['base_height'].compute(context), torch.full((2,1),.32), atol=1e-6, rtol=0)
        feet = obs['foot_height'].compute(context)
        torch.testing.assert_close(feet[0],feet[1],atol=1e-6,rtol=0)
        reward = env.task.reward_manager.terms['base_height_l2'].compute(context)
        torch.testing.assert_close(reward[0],reward[1],atol=1e-6,rtol=0)
        assert not env.task.check_terminated(context)[0].any()
    finally:
        env.close()


def test_one_batched_query_is_shared_by_all_height_terms_and_subsets(env, monkeypatch):
    calls=[]
    query=env.simulator.get_ground_heights
    def sample(xy, env_ids=None):
        assert env_ids is not None
        calls.append((xy.shape, env_ids.tolist()))
        return query(xy, env_ids)
    monkeypatch.setattr(env.simulator,'get_ground_heights',sample)
    context=task_context(env,torch.tensor([2,0]))
    assert calls == [(torch.Size([2,5,2]),[2,0])]
    observation,_=env.task.compute_observation(context,torch.tensor([2,0]))
    assert observation.shape == (2,112)
    assert len(calls)==1
    assert context.state.foot_ground_heights.shape == (2,4)


@pytest.mark.parametrize('x', [2.3,1.95])
def test_root_or_foot_outside_patch_terminates_only_selected_environment(env, monkeypatch, x):
    term=env.task.randomization_manager.terms['terrain']
    patches, generations = term.patches, term.assignments.generations.copy()
    env.simulator.set_root_positions(torch.tensor([1]),torch.tensor([[x,0.,.32]]))
    query=env.simulator.get_ground_heights
    def sample(xy, env_ids=None):
        assert torch.isfinite(xy).all()
        return query(xy,env_ids)
    monkeypatch.setattr(env.simulator,'get_ground_heights',sample)
    obs,transition,reward,terminated,truncated,info=env.step(torch.zeros(3,12))
    assert info['termination/terrain_bounds'].tolist()==[False,True,False]
    assert terminated.tolist()==[False,True,False]
    assert env.current_episode_steps.tolist()==[1,0,1]
    assert info['state/ground_valid'].all()
    assert info['state/terrain_in_bounds'].tolist()==[True,False,True]
    selected_state=env.simulator.get_state(torch.tensor([1]))
    assert selected_state.ground_valid is not None and selected_state.ground_valid.all()
    assert info['state/ground_valid'].all()
    assert info['state/terrain_in_bounds'].tolist()==[True,False,True]
    assert torch.isfinite(obs).all() and torch.isfinite(transition).all() and torch.isfinite(reward).all()
    assert term.assignments.generations == generations
    assert all(a is b for a,b in zip(term.patches,patches))
    assert task_context(env).state.ground_valid.all()
    assert info['reward/base_height_l2'][1] < 0


def test_real_trunk_contact_detected_on_all_three_surfaces(env):
    positions = torch.tensor([[1.2, 0., .32]]).repeat(3, 1)
    ground = env.simulator.get_ground_heights(positions[:, None, :2])[:, 0]
    positions[:, 2] = ground + .02
    env.simulator.set_root_positions(torch.arange(3), positions)
    context=task_context(env)
    assert BodyContact(context=env.context,model_context=env.simulator.model_context,
                       body_names=['trunk']).compute(context).all()
    penalty=IllegalContactL1(context=env.context,model_context=env.simulator.model_context,num_envs=3).compute(context)
    assert (penalty>0).all()


def test_task_context_is_complete_at_construction(env, monkeypatch):
    from envs.tasks.utils import TaskContext
    constructed = []

    def construct_context(**values):
        assert values['state'].base_ground_height.shape == (3,)
        assert not {'base_ground_height','foot_ground_heights','ground_valid'} & values.keys()
        assert values['state'].foot_ground_heights.shape == (3,4)
        assert values['state'].ground_valid.all()
        context = TaskContext(**values)
        constructed.append(context)
        return context

    monkeypatch.setattr('envs.tasks.base.TaskContext', construct_context)
    context = task_context(env)
    assert constructed == [context]
    assert not hasattr(env.task, "_simulator")


@pytest.mark.parametrize('env_ids', [None, torch.tensor([2,0]), torch.empty(0,dtype=torch.long)])
def test_ground_bounds_describe_selected_world_regions(env, env_ids):
    bounds = env.simulator.get_ground_bounds(env_ids)
    assert bounds is not None
    count = env.num_envs if env_ids is None else env_ids.numel()
    lower, upper = bounds
    torch.testing.assert_close(lower,torch.full((count,2),-2.))
    torch.testing.assert_close(upper,torch.full((count,2),2.))
    assert lower.device == env.context.device and lower.dtype == env.context.dtype


@pytest.mark.parametrize('x', [2.3, 1.95])
def test_without_terrain_bounds_term_outside_patch_keeps_running(runtime_context, tmp_path, x):
    config = terrain_task_config()
    config['termination_manager_config']['terms'].pop('terrain_bounds', None)
    path = tmp_path/'task.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    env = terrain_environment(runtime_context, path, 1)
    try:
        env.simulator.set_root_positions(torch.tensor([0]), torch.tensor([[x, 0., .32]]))
        context = task_context(env)
        assert context.state.ground_valid is not None and context.state.terrain_in_bounds is not None
        assert context.state.ground_valid.all()
        assert not context.state.terrain_in_bounds.any()
        assert not env.task.check_terminated(context)[0].any()
        for _ in range(3):
            observation, _, reward, terminated, _, info = env.step(torch.zeros(1,12))
            assert not terminated.any()
            assert 'termination/terrain_bounds' not in info
            assert info['state/ground_valid'].all()
            assert torch.isfinite(observation).all() and torch.isfinite(reward).all()
        assert env.current_episode_steps.item() == 3
        if x > 2:
            torch.testing.assert_close(context.state.base_ground_height, torch.zeros(1))
        else:
            # Inside and outside points are queried independently in one row.
            assert context.state.foot_ground_heights is not None
            positions = context.state.geom_xpos[:, env.simulator.model_context.foot_geom_ids, :2]
            outside = positions[0,:,0] > 2
            assert outside.any() and not outside.all()
            assert not context.state.foot_ground_heights[0,outside].any()
    finally:
        env.close()


def test_nonfinite_or_beyond_peripheral_ground_has_invalid_signals(env):
    env.simulator.set_root_positions(torch.tensor([0]),torch.tensor([[1100.,0.,.32]]))
    state = env.simulator.get_state()
    assert state.ground_valid is not None and state.terrain_in_bounds is not None
    assert state.ground_valid.tolist() == [False,True,True]
    assert torch.isfinite(state.base_ground_height).all()
    assert not state.terrain_in_bounds[0]


def test_peripheral_ground_renders_with_real_opengl(env):
    env.simulator.set_root_positions(torch.tensor([0]),torch.tensor([[3.,0.,.32]]))
    image = env.simulator.render()
    assert image is not None and image.ndim == 3
    assert image.dtype == np.uint8 and image.std() > 1


def test_existing_anymal_phase_config_runs_mixed_terrain_and_partial_reset(runtime_context):
    env = VectorEnv(replace(runtime_context,load_dir=ROOT,save_dir=ROOT))
    component = Component(None,None,None,None,
        ComponentInfo('mujoco',ROOT/'configs/simulators/mujoco_anymal_c_rgb_array.yaml'),
        ComponentInfo('locomotion',ROOT/'configs/tasks/locomotion_anymal_c_phase.yaml'))
    try:
        env.config_update(component,num_envs=3,max_episode_steps=100)
        observation = env.reset()
        assert observation.shape == (3,120)
        terrain = cast(TerrainRandomizationTerm,env.task.randomization_manager.terms['terrain'])
        assert [patch.terrain_type for patch in terrain.patches if patch is not None] == ['plane','slope','rough']
        assert terrain.config.spawn_size == (1.5,1.2)
        seen = torch.zeros(3,4,dtype=torch.bool)
        for _ in range(30):
            observation,_,reward,terminated,_,info = env.step(torch.zeros(3,12))
            assert torch.isfinite(observation).all() and torch.isfinite(reward).all()
            assert not terminated.any()
            assert info['state/ground_valid'].all()
            seen |= info['state/foot_ground_contact']
        assert seen.all()
        patches = terrain.patches
        env.current_episode_steps[1] = 99
        env.step(torch.zeros(3,12))
        assert env.current_episode_steps.tolist() == [31,0,31]
        assert all(a is b for a,b in zip(patches,terrain.patches))
    finally:
        env.close()
