"""Real MuJoCo phase-4 PPO, checkpoint, geometry reconstruction and fixed playback."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envs.vector_env import VectorEnv
from utils import ComponentInfo, RuntimeContext, load_yaml
from verify_command_curriculum import build_component, configure, VerifyFinite


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_equal(a, b)
    else:
        assert first == second


def component_for(output, mode, friction=False):
    component = build_component(output)
    config = load_yaml(component.task.config)
    terrain = config['randomization_manager_config']['terms']['terrain']
    terrain['resolution'] = .2
    terrain['difficulty']['params']['values'] = [.25, 1.]
    for axis in ('category', 'difficulty'):
        terrain[axis]['type'] = 'Weighted' if axis == 'category' else 'Uniform'
        terrain[axis]['params'].pop('curriculum', None)
    # Small candidate spaces exercise updates without an expensive long curriculum.
    for command in config['command_manager_config']['terms'].values():
        command['params']['num_cell'] = 2
    if mode == 'uniform':
        config['curriculum_manager_config'] = {}
        for command in config['command_manager_config']['terms'].values():
            command['type'] = 'Uniform'
            command['params'].pop('curriculum', None)
            command['params'].pop('num_cell', None)
    else:
        for axis in ('category', 'difficulty'):
            terrain[axis]['type'] = 'Curriculum'
            terrain[axis]['params']['curriculum'] = 'locomotion'
        if mode == 'terrain_only':
            for command in config['command_manager_config']['terms'].values():
                command['type'] = 'Uniform'
                command['params'].pop('curriculum', None)
                command['params'].pop('num_cell', None)
        elif mode == 'separate':
            config['curriculum_manager_config']['terms']['terrain_course'] = deepcopy(
                config['curriculum_manager_config']['terms']['locomotion'])
            for axis in (terrain['category'], terrain['difficulty']):
                axis['params']['curriculum'] = 'terrain_course'
        if mode == 'difficulty_only':
            terrain['category']['type'] = 'Weighted'
            terrain['category']['params'].pop('curriculum')
        elif mode == 'type_only':
            terrain['difficulty']['type'] = 'Fixed'
            terrain['difficulty']['params'].pop('curriculum')
            terrain['difficulty']['params']['values'] = [1.]
        elif mode == 'axis_separate':
            config['curriculum_manager_config']['terms']['levels'] = deepcopy(
                config['curriculum_manager_config']['terms']['locomotion'])
            terrain['difficulty']['params']['curriculum'] = 'levels'
        for course in config['curriculum_manager_config']['terms'].values():
            course['params']['min_samples_per_cell'] = 1
    if friction:
        strategy = 'Uniform' if mode == 'uniform' else 'Curriculum'
        params = {'factors': [.2, .8, 1.1]}
        if strategy == 'Curriculum':
            params['curriculum'] = 'locomotion'
        config['randomization_manager_config']['terms']['friction'] = {
            'type': strategy, 'seed': 17, 'params': params,
        }
    path = output / 'task.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return replace(component, task=ComponentInfo('locomotion', path))


def get_terrain(environment):
    return environment.task.randomization_manager.terms['terrain']


def verify_mode(output, mode, render, friction=False):
    component = component_for(output, mode, friction)
    context = RuntimeContext(torch.device('cpu'), torch.float32, 4, 0, True, ROOT, output)
    torch.manual_seed(0)
    source = configure(context, component, 2)
    resumed = None
    physical = None
    playback = None
    result = {'passed': False, 'render_requested': render}
    try:
        checker = VerifyFinite(source)
        source.callbacks.append(checker)
        source.train()
        assert checker.updates == 2
        env = source.environment
        term = get_terrain(env)
        term.request_reallocation(torch.tensor([2]))
        term.request_spawn(torch.tensor([1]))
        before = deepcopy(env.checkpoint_state_dict())
        patches = term.patches
        heights = [model.hfield_data.copy() for model in env.simulator.models]
        checkpoint = source.save(output / 'terrain.pt')
        source.test(num_episodes=4)
        assert_equal(before, env.checkpoint_state_dict())

        physical = VectorEnv(context)
        physical.config_update(component, num_envs=4, max_episode_steps=32)
        physical.load_checkpoint_state_dict(before)
        physical.reset()
        assert_equal(before, physical.checkpoint_state_dict())
        assert get_terrain(physical).patches == patches
        for model, saved in zip(physical.simulator.models, heights):
            np.testing.assert_array_equal(model.hfield_data, saved)
        if friction:
            for model, original in zip(physical.simulator.models, env.simulator.models):
                np.testing.assert_array_equal(model.geom_friction, original.geom_friction)

        # The actual temporary single-environment construction path, without a viewer.
        playback = physical._create_playback_environment()
        playback.reset()
        assert get_terrain(playback).patches[0] == patches[physical.playback_env_index]
        fixed = deepcopy(playback.checkpoint_state_dict())
        for _ in range(3):
            playback.reset()
            assert_equal(fixed, playback.checkpoint_state_dict())
        if friction:
            np.testing.assert_array_equal(
                playback.simulator.models[0].geom_friction,
                physical.simulator.models[physical.playback_env_index].geom_friction,
            )
        assert_equal(before, physical.checkpoint_state_dict())

        # Restoration reset above must not advance either independent or joint conditions.
        env.set_training_mode(True)
        env.load_checkpoint_state_dict(before)
        env.reset()
        for _ in range(4):
            env.reset()
            physical.reset()
            assert_equal(env.checkpoint_state_dict(), physical.checkpoint_state_dict())
            assert get_terrain(env).patches == get_terrain(physical).patches
            if friction:
                for model, original in zip(physical.simulator.models, env.simulator.models):
                    np.testing.assert_array_equal(model.geom_friction, original.geom_friction)

        if render:
            from PIL import Image
            playback.simulator.prepare_playback('rgb_array')
            image = playback.simulator.render()
            assert image is not None and image.std() > 1
            Image.fromarray(image).save(output / 'playback.png')

        resumed = configure(context, component, 4, checkpoint)
        assert_equal(before, resumed.environment.checkpoint_state_dict())
        checker = VerifyFinite(resumed)
        resumed.callbacks.append(checker)
        parameters = [p.detach().clone() for p in resumed.algorithm.policy.parameters()]
        resumed.train()
        delta = sum(float((p.detach() - old).square().sum())
                    for p, old in zip(resumed.algorithm.policy.parameters(), parameters)) ** .5
        assert checker.updates == 2 and resumed.current_iteration == 3 and delta > 0
        result.update(passed=True, initial_updates=2, resumed_updates=2,
                      transitions=1024, finite_signals=True, parameter_delta_l2=delta,
                      evaluation_frozen=True, exact_height_buffer_reconstruction=True,
                      four_future_assignments_equal=True, single_environment_playback_fixed=True,
                      render_verified=render,
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      sampled_types=sorted({p.terrain_type for p in patches}))
        if friction:
            result['friction_buffers_verified'] = True
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for resource in (playback, physical, resumed, source):
            if resource is not None:
                resource.close()
        result['cleanup_returned'] = True
        (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'temp/stage4_mujoco')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--friction', action='store_true',
                        help='Verify both independent and joint friction sampling with terrain.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    modes = (('uniform', 'joint') if args.friction else
             ('uniform', 'terrain_only', 'joint', 'separate', 'difficulty_only', 'type_only', 'axis_separate'))
    report = {mode: verify_mode(args.output_dir.resolve() / mode, mode, args.render, args.friction)
              for mode in modes}
    (args.output_dir / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('TERRAIN CURRICULUM VERIFICATION PASSED', flush=True)


if __name__ == '__main__':
    main()
