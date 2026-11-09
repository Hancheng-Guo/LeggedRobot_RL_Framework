"""Backend-independent height semantics; existing world-coordinate terms stay explicit."""
from dataclasses import replace
import pytest
import torch
from envs.simulators.utils import SimulatorState
from envs.tasks.utils import TaskContext
from envs.tasks.managers.observation.terms.state import BaseHeight as HeightObservation
from envs.tasks.managers.observation.terms.foot import FootHeight
from envs.tasks.managers.reward.terms.state import BaseHeightL2, BaseHeightL2XySpeedWeighted
from envs.tasks.managers.reward.terms.foot import FootLiftHeightDiffCommandWeightedExp, FootLiftHeightDiffCommandGatedL2
from envs.tasks.managers.termination.terms.state import BaseHeight as HeightTermination
from envs.tasks.managers.reward.terms.contact import IllegalContactL1


def height_context(model_context, *, relative=True):
    offsets = torch.tensor([0., .7])
    qpos = torch.zeros(2, model_context.nq)
    qpos[:, model_context.base_pos_qpos_ids[2]] = .15 + offsets
    geoms = torch.zeros(2, len(model_context.geom_names), 3)
    geoms[:, model_context.foot_geom_ids, 2] = (.12 + offsets)[:, None]
    state = SimulatorState(qpos=qpos, qvel=torch.ones(2, model_context.nv),
        qacc=torch.zeros(2, model_context.nv), ctrl=torch.zeros(2, model_context.nu),
        geom_xpos=geoms, geom_xvel=torch.zeros(2, len(model_context.geom_names), 6),
        actuator_force=torch.zeros(2, model_context.nu), base_lin_vel_body=torch.zeros(2,3),
        base_ang_vel_body=torch.zeros(2,3), contact_geom_ids=torch.empty(2,0,2,dtype=torch.long),
        contact_forces=torch.empty(2,0,6), foot_ground_contact=torch.zeros(2,1,dtype=torch.bool),
        foot_contact_normal_force=torch.zeros(2,1),
        base_ground_height=offsets if relative else None,
        foot_ground_heights=offsets[:,None] if relative else None,
        ground_valid=torch.ones(2,dtype=torch.bool) if relative else None,
        terrain_in_bounds=torch.ones(2,dtype=torch.bool) if relative else None)
    return TaskContext(state=state, command={k:torch.ones(2,1) for k in ('lin_vel_x','lin_vel_y','ang_vel_z')},
        last_command={}, action=torch.zeros(2,2), last_action=torch.zeros(2,2),
        episode_step=torch.zeros(2,dtype=torch.long), step_dt=.02)


HEIGHT_TERMS = [
    (HeightObservation, {}), (FootHeight, {}),
    (BaseHeightL2, {'target_height':.3}), (BaseHeightL2XySpeedWeighted, {'target_height':.3}),
    (FootLiftHeightDiffCommandWeightedExp, {'target_height':.1}),
    (FootLiftHeightDiffCommandGatedL2, {'target_height':.1}),
    (HeightTermination, {'min_height':.2}),
]


@pytest.mark.parametrize('cls,params', HEIGHT_TERMS)
def test_ground_height_signals_are_translation_invariant(runtime_context, model_context, cls, params):
    context = height_context(model_context)
    term = cls(context=runtime_context, model_context=model_context, **params)
    value = term.compute(context)
    torch.testing.assert_close(value[0], value[1], rtol=1e-5, atol=1e-6)
    world = cls(context=runtime_context, model_context=model_context, height_reference='world', **params)
    world_value = world.compute(context)
    assert not torch.allclose(world_value[0], world_value[1])


@pytest.mark.parametrize('cls,params', HEIGHT_TERMS)
def test_flat_fallback_and_zero_ground_are_exactly_compatible(runtime_context, model_context, cls, params):
    context = height_context(model_context, relative=False)
    term = cls(context=runtime_context, model_context=model_context, **params)
    old_value = term.compute(context)
    sampled = replace(context, state=replace(context.state, base_ground_height=torch.zeros(2),
                      foot_ground_heights=torch.zeros(2,1), ground_valid=torch.ones(2,dtype=torch.bool)))
    torch.testing.assert_close(term.compute(sampled), old_value, rtol=0, atol=0)


@pytest.mark.parametrize('cls,params', HEIGHT_TERMS)
def test_invalid_height_reference_rejected(runtime_context, model_context, cls, params):
    with pytest.raises(ValueError, match='height_reference'):
        cls(context=runtime_context, model_context=model_context, height_reference='invalid', **params)


@pytest.mark.parametrize('cls,params', HEIGHT_TERMS)
def test_invalid_ground_signals_are_masked_without_fake_height_rewards(runtime_context, model_context, cls, params):
    context = height_context(model_context)
    context.state.ground_valid = torch.tensor([True, False])
    value = cls(context=runtime_context, model_context=model_context, **params).compute(context)
    assert torch.count_nonzero(value[1]) == 0
    assert torch.isfinite(value).all()


def test_empty_contact_reward_uses_selected_batch_size(runtime_context, model_context):
    context = height_context(model_context)
    context = replace(context, state=context.state.select(torch.tensor([1])), env_ids=torch.tensor([1]))
    term = IllegalContactL1(context=runtime_context, model_context=model_context, num_envs=2)
    assert term.compute(context).shape == (1,)


def test_manual_training_components_reuse_existing_configs(tmp_path):
    import importlib.util
    from pathlib import Path
    from utils import load_yaml
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("terrain_training_acceptance",
        root/'tests/manual/verify_mujoco_terrain_training.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    component = module.build_training_component(tmp_path)
    assert component.runner is None
    assert all(getattr(component, key).config.is_file()
               for key in ('algorithm','policy','environment','simulator','task'))
    assert load_yaml(component.environment.config) == {'num_envs':8,'max_episode_steps':200}
    source = load_yaml(root/'configs/tasks/locomotion_anymal_c_phase.yaml')
    diagnostic = load_yaml(component.task.config)
    assert list(diagnostic['randomization_manager_config']['terms']['terrain']['types']) == list(
        source['randomization_manager_config']['terms']['terrain']['types'])
    assert list(diagnostic['observation_manager_config']['terms']) == list(
        source['observation_manager_config']['terms'])
    moving = module.build_training_component(tmp_path/'moving', .3)
    params = load_yaml(moving.task.config)['command_manager_config']['terms']['lin_vel_x']['params']
    assert params == {'min_value':.3,'max_value':.3}


def test_manual_training_checker_does_not_require_optional_terrain_termination(model_context):
    import runpy
    from pathlib import Path
    from types import SimpleNamespace
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               'tests/manual/verify_mujoco_terrain_training.py'))
    model_context = replace(model_context, foot_geom_ids=torch.arange(4))
    calls = []

    def sample(xy, env_ids):
        calls.append(env_ids.tolist())
        return torch.full(xy.shape[:2], .02)

    environment = SimpleNamespace(num_envs=2, current_episode_steps=torch.ones(2),
        simulator=SimpleNamespace(model_context=model_context, get_ground_heights=sample),
        task=SimpleNamespace(randomization_manager=SimpleNamespace(terms={
            'terrain':SimpleNamespace(patches=[SimpleNamespace(terrain_type='rough')]*2)})))
    checker = module['VerifySignals'](SimpleNamespace(environment=environment))
    info = {
        'reward/base_height_l2':torch.zeros(2),
        'state/foot_ground_contact':torch.zeros(2,4,dtype=torch.bool),
        'state/qpos':torch.zeros(2,model_context.nq),
        'state/geom_xpos':torch.zeros(2,4,3),
        'state/ground_valid':torch.tensor([True,False]),
    }
    assert checker._on_step_end(info)
    assert calls == [[0]] and checker.nonflat_samples['rough'] == 1


def test_simulator_queries_per_environment_physical_bounds(runtime_context, model_context):
    from typing import cast
    from envs.simulators.base import BaseSimulator
    context = height_context(model_context)
    origins = torch.tensor([[10.,0.],[20.,0.]])
    context.state.qpos[:, model_context.base_pos_qpos_ids[:2]] = origins
    context.state.geom_xpos[:, model_context.foot_geom_ids, :2] = origins[:,None]
    calls = []

    class GroundBackend:
        def get_ground_bounds(self, env_ids=None):
            selected = slice(None) if env_ids is None else env_ids
            return origins[selected]-1, origins[selected]+1

        def get_ground_heights(self, xy, env_ids=None):
            assert env_ids is not None
            calls.append(env_ids.tolist())
            return torch.tensor([0.,.7])[env_ids,None].expand(xy.shape[:2])

    backend = cast(BaseSimulator, GroundBackend())
    backend.num_envs = 2
    backend.model_context = model_context
    env_ids_tensor = torch.tensor([1,0])
    selected = context.state.select(env_ids_tensor)
    data = BaseSimulator._get_ground_state(backend, selected.qpos, selected.geom_xpos, env_ids_tensor)
    assert calls == [[1,0]]
    torch.testing.assert_close(data['base_ground_height'],torch.tensor([.7,0.]))
    assert data['ground_valid'].all()
    # A foot can leave its assigned region while its root remains inside.
    context.state.geom_xpos[1,model_context.foot_geom_ids,0] = 22.
    data = BaseSimulator._get_ground_state(backend, context.state.qpos, context.state.geom_xpos)
    assert data['ground_valid'].all()
    assert data['terrain_in_bounds'].tolist() == [True,False]
    assert calls[-1] == [0,1]
    # Non-finite coordinates do not enter the physical height query.
    context.state.qpos[1,model_context.base_pos_qpos_ids[0]] = float('nan')
    data = BaseSimulator._get_ground_state(backend, context.state.qpos, context.state.geom_xpos)
    assert calls[-1] == [0]
    assert data['ground_valid'].tolist() == [True,False]
    assert torch.isfinite(data['base_ground_height']).all()


def test_unbounded_ground_preserves_legacy_signals_without_height_query(runtime_context, model_context):
    from typing import cast
    from envs.simulators.base import BaseSimulator

    class GroundBackend:
        def get_ground_bounds(self, env_ids=None):
            return None

        def get_ground_heights(self, *args):
            raise AssertionError('Infinite plane does not require sampling')

    backend = cast(BaseSimulator, GroundBackend())
    state = height_context(model_context).state
    assert BaseSimulator._get_ground_state(backend, state.qpos, state.geom_xpos) == {}


@pytest.mark.parametrize('relative', [False, True])
def test_simulator_state_optional_ground_data_clone_and_select(model_context, relative):
    state = height_context(model_context, relative=relative).state
    cloned = state.clone()
    selected = state.select(torch.tensor([1,0]))
    empty = state.select(torch.empty(0,dtype=torch.long))
    names = ['base_ground_height','foot_ground_heights','ground_valid','terrain_in_bounds']
    for name in names:
        value = getattr(state,name)
        if relative:
            assert name in state.as_dict()
            torch.testing.assert_close(getattr(selected,name),value[[1,0]])
            assert getattr(empty,name).shape[0] == 0
            value[0] = 0
            assert getattr(cloned,name).data_ptr() != value.data_ptr()
            assert getattr(selected,name).data_ptr() != value.data_ptr()
        else:
            assert name not in state.as_dict()
            assert getattr(cloned,name) is None and getattr(selected,name) is None
    assert empty.qpos.shape[0] == 0


@pytest.mark.parametrize('in_bounds', [None, [True, False], [False, True]])
def test_terrain_bounds_term_is_optional_and_uses_region_not_ground_valid(runtime_context, model_context, in_bounds):
    from envs.tasks.managers.termination.terms.terrain import TerrainBounds
    from envs.tasks.managers.termination import TerminationManager
    context = height_context(model_context)
    context.state.terrain_in_bounds = None if in_bounds is None else torch.tensor(in_bounds)
    context.state.ground_valid = torch.tensor([False, True])
    expected = torch.zeros(2, dtype=torch.bool) if in_bounds is None else ~torch.tensor(in_bounds)
    term = TerrainBounds(context=runtime_context)
    torch.testing.assert_close(term.compute(context), expected)
    manager = TerminationManager(2, runtime_context, model_context, terms={})
    terminated, info = manager.compute(context)
    assert not terminated.any() and not info
    selected = replace(context, state=context.state.select(torch.tensor([1])))
    torch.testing.assert_close(term.compute(selected), expected[[1]])
    empty = replace(context, state=context.state.select(torch.empty(0, dtype=torch.long)))
    assert term.compute(empty).shape == (0,)
