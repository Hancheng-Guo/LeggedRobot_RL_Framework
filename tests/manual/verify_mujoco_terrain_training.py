"""Stage-3 real PPO smoke: finite signals, actual nonflat starts, resets and updates."""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from collections import Counter
import yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from runners.on_policy import OnPolicyRunner
from runners.callbacks.base import BaseCallback
from utils import RuntimeContext, Component, ComponentInfo, load_yaml
from envs.tasks.utils import base_height


ENVIRONMENT_CONFIG = {"num_envs": 8, "max_episode_steps": 200}
RUNNER_CONFIG = {"rollout_length": 128, "rollout_length_history_size": 50, "callbacks": ["logging"]}


def build_training_component(output_dir: Path, command_speed: float = 0.) -> Component:
    """Keep acceptance-only parameters outside the business config directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_path = output_dir.resolve() / "environment.yaml"
    environment_path.write_text(yaml.safe_dump(ENVIRONMENT_CONFIG), encoding="utf-8")
    task_config = load_yaml(ROOT / "configs/tasks/locomotion_anymal_c_phase.yaml")
    # Fixed diagnostic commands live only in the generated acceptance config.
    for name, value in (("lin_vel_x", command_speed), ("lin_vel_y", 0.), ("ang_vel_z", 0.)):
        task_config["command_manager_config"]["terms"][name] = {
            "type":"UniformOnReset", "params":{"min_value":value,"max_value":value}}
    task_config["curriculum_manager_config"] = {"terms":{}}
    task_path = output_dir.resolve() / ("moving_task.yaml" if command_speed else "standing_task.yaml")
    task_path.write_text(yaml.safe_dump(task_config, sort_keys=False), encoding="utf-8")
    return Component(
        runner=None,
        algorithm=ComponentInfo("ppo", ROOT / "configs/algorithms/ppo_num_batch_4.yaml"),
        policy=ComponentInfo("actor_critic", ROOT / "configs/policies/actor_critic_direct_actuation.yaml"),
        environment=ComponentInfo("vector_env", environment_path),
        simulator=ComponentInfo("mujoco", ROOT / "configs/simulators/mujoco_anymal_c_rgb_array.yaml"),
        task=ComponentInfo("locomotion", task_path),
    )


class VerifySignals(BaseCallback):
    def __init__(self, runner):
        super().__init__(runner)
        self.steps = 0
        self.episodes = []
        self.lengths = torch.zeros(runner.environment.num_envs, dtype=torch.long)
        self.terminations = Counter()
        self.nonflat_samples = Counter()
        self.updates = []
        self.height_reward_range = [float("inf"), float("-inf")]
        self.contacts = torch.zeros(runner.environment.num_envs, 4, dtype=torch.bool)

    def _on_step_end(self, info, **kwargs):
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), key
        height_reward = info['reward/base_height_l2']
        self.height_reward_range[0] = min(self.height_reward_range[0], float(height_reward.min()))
        self.height_reward_range[1] = max(self.height_reward_range[1], float(height_reward.max()))
        env = self.runner.environment
        self.steps += 1
        self.lengths += 1
        done = env.current_episode_steps == 0
        self.episodes.extend(self.lengths[done].tolist())
        self.lengths[done] = 0
        for key, value in info.items():
            if key.startswith('termination/'):
                self.terminations[key] += int(value.sum())
        self.contacts |= info['state/foot_ground_contact']
        model = env.simulator.model_context
        xy = torch.cat((info['state/qpos'][:,model.base_pos_qpos_ids[:2]][:,None],
                        info['state/geom_xpos'][:,model.foot_geom_ids,:2]), dim=1)
        valid = info['state/ground_valid']
        if valid.any():
            ids = torch.arange(env.num_envs)[valid]
            heights = env.simulator.get_ground_heights(xy[valid], ids)
            patches = env.task.randomization_manager.terms['terrain'].patches
            for env_id, nonflat in zip(ids.tolist(), (heights.abs()>1e-5).any(dim=1).tolist()):
                if nonflat:
                    self.nonflat_samples[patches[env_id].terrain_type] += 1
        return True

    def _on_iteration_end(self, info, **kwargs):
        row = {key:float(value) for key,value in info.items() if isinstance(value,(float,int))}
        assert all(math.isfinite(value) for value in row.values()), row
        assert all(torch.isfinite(p).all() for p in self.runner.algorithm.policy.parameters())
        self.updates.append(row)
        print(f'PPO iteration {self.runner.current_iteration}: loss={row["rollout/loss"]:.5g}', flush=True)
        return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'tests/manual/results/stage3_mujoco')
    parser.add_argument('--iterations',type=int,default=16)
    parser.add_argument('--command-speed',type=float,default=0.,help='Optional diagnostic forward command (m/s).')
    args=parser.parse_args()
    if not math.isfinite(args.command_speed) or abs(args.command_speed)>1:
        parser.error('--command-speed must be finite and within [-1,1]')
    if args.iterations<4:
        parser.error('--iterations must be >= 4')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    context=RuntimeContext(torch.device('cpu'),torch.float32,4,0,True,ROOT,args.output_dir.resolve())
    component=build_training_component(args.output_dir, args.command_speed)
    runner=OnPolicyRunner(context)
    report={'script':str(Path(__file__).resolve()),'seed':0,'iterations':args.iterations,
            'environment_config':dict(ENVIRONMENT_CONFIG),'runner_config':dict(RUNNER_CONFIG),
            'component_configs':{name:str(getattr(component,name).config)
                                 for name in ('algorithm','policy','environment','simulator','task')},
            'measured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'nonflat_start_is_diagnostic_only':True,'command_speed':args.command_speed}
    started=time.perf_counter()
    try:
        runner.config_update(component,max_iterations=args.iterations,stage_index=0,
                             **RUNNER_CONFIG)
        env=runner.environment
        original_reset=env.reset
        def reset_on_surfaces():
            original_reset()
            terrain=env.task.randomization_manager.terms['terrain']
            ids=torch.tensor([i for i,p in enumerate(terrain.patches) if p.terrain_type!='plane'])
            assert len(ids)>=2
            reset_height=env.simulator.get_state(ids).qpos[:,2].clone()
            positions=torch.tensor([[1.2,0.,0.]]).repeat(len(ids),1)
            positions[:,2]=reset_height
            env.simulator.set_root_positions(ids,positions)
            state=env.simulator.get_state(ids)
            foot_xy=state.geom_xpos[:,env.simulator.model_context.foot_geom_ids,:2]
            ground=env.simulator.get_ground_heights(foot_xy,ids)
            positions[:,2]=reset_height+ground.max(dim=1).values
            env.simulator.set_root_positions(ids,positions)
            # Refresh task histories and initial observation after the diagnostic
            # placement. Ordinary episode resets retain the normal safe spawn.
            env.task.reset()
            context=env.task.build_task_context(env.simulator.get_state(),env.current_episode_steps,
                                                env.simulator.control_dt)
            obs,_=env.task.compute_observation(context)
            assert torch.isfinite(obs).all() and context.state.ground_valid.all()
            report['initial_nonflat_env_ids']=ids.tolist()
            report['initial_foot_ground_heights']=ground.tolist()
            report['initial_relative_base_height']=base_height(context, int(env.simulator.model_context.base_pos_qpos_ids[2])).tolist()
            return obs
        env.reset=reset_on_surfaces
        checker=VerifySignals(runner)
        runner.callbacks.append(checker)
        before=[p.detach().clone() for p in runner.algorithm.policy.parameters()]
        result=runner.train()
        parameter_delta=sum(float((p.detach()-old).square().sum()) for p,old in zip(runner.algorithm.policy.parameters(),before))**.5
        assert result.current_iteration==args.iterations-1
        assert len(checker.updates)==args.iterations and parameter_delta>0
        assert checker.episodes and min(checker.episodes)>1, checker.episodes
        assert checker.nonflat_samples['slope']>0 and checker.nonflat_samples['rough']>0
        assert checker.contacts.all()
        if args.command_speed:
            assert max(abs(v) for v in checker.height_reward_range)>1e-6
        report.update(passed=True,elapsed_seconds=time.perf_counter()-started,num_envs=env.num_envs,
            rollout_length=runner.rollout_length,transitions=checker.steps*env.num_envs,
            completed_episodes=len(checker.episodes),min_episode_length=min(checker.episodes),
            mean_episode_length=sum(checker.episodes)/len(checker.episodes),
            termination_counts=dict(checker.terminations),nonflat_samples=dict(checker.nonflat_samples),
            contacts_seen=checker.contacts.tolist(),parameter_delta_l2=parameter_delta,
            updates=checker.updates,observation_dim=env.observation_dim,
            base_height_reward_range=checker.height_reward_range)
    finally:
        runner.close()
        report['cleanup_returned']=True
        (args.output_dir/'training_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('STAGE 3 PPO VERIFICATION PASSED',flush=True)


if __name__=='__main__':
    main()
