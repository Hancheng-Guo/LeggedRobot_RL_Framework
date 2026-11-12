"""Real command-course PPO, evaluation freeze, resume, with explicit curriculum references."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runners import OnPolicyRunner
from runners.callbacks.base import BaseCallback
from utils import Component, ComponentInfo, RuntimeContext, load_yaml


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_equal(first[name], second[name])
    else:
        assert first == second


def build_component(output):
    output.mkdir(parents=True, exist_ok=True)
    environment = output / "environment.yaml"
    environment.write_text(yaml.safe_dump({"num_envs": 4, "max_episode_steps": 32}), encoding="utf-8")
    simulator = output / "simulator.yaml"
    simulator_config = load_yaml(ROOT / "configs/simulators/mujoco_anymal_c_rgb_array.yaml")
    simulator_config.pop("render_mode", None)
    simulator.write_text(yaml.safe_dump(simulator_config, sort_keys=False), encoding="utf-8")
    return Component(
        runner=None,
        algorithm=ComponentInfo("ppo", ROOT / "configs/algorithms/ppo_num_batch_4.yaml"),
        policy=ComponentInfo("actor_critic", ROOT / "configs/policies/actor_critic_direct_actuation.yaml"),
        environment=ComponentInfo("vector_env", environment),
        simulator=ComponentInfo("mujoco", simulator),
        task=ComponentInfo("locomotion", ROOT / "configs/tasks/locomotion_anymal_c_phase.yaml"),
    )


class VerifyFinite(BaseCallback):
    def __init__(self, runner):
        super().__init__(runner)
        self.updates = 0
        self.steps = 0

    def _on_step_end(self, info, **kwargs):
        self.steps += 1
        for name, value in info.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), name
        return True

    def _on_iteration_end(self, info, **kwargs):
        self.updates += 1
        assert all(math.isfinite(value) for value in info.values() if isinstance(value, (int, float)))
        assert all(torch.isfinite(p).all() for p in self.runner.algorithm.policy.parameters())
        return True


def configure(context, component, iterations, checkpoint=None):
    runner = OnPolicyRunner(context)
    try:
        if checkpoint is not None:
            runner.prepare_checkpoint_load(checkpoint)
        runner.config_update(component, max_iterations=iterations, rollout_length=64,
                             callbacks=["logging"], stage_index=0)
        if checkpoint is not None:
            runner.load()
        return runner
    except BaseException:
        runner.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "temp/command_curriculum")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    component = build_component(output)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    context = RuntimeContext(torch.device("cpu"), torch.float32, 4, 0, True, ROOT, output)
    source = configure(context, component, 2)
    resumed = None
    report = {"passed": False, "num_envs": 4, "rollout_length": 64, "episode_steps": 32}
    try:
        checker = VerifyFinite(source)
        source.callbacks.append(checker)
        source.train()
        assert checker.updates == 2
        curriculum = source.environment.task.curriculum_manager
        before = deepcopy(curriculum.checkpoint_state_dict())
        checkpoint = source.save(output / "command.pt")
        source.test(num_episodes=4)
        assert_equal(before, curriculum.checkpoint_state_dict())
        report["evaluation_statistics_frozen"] = True
        resumed = configure(context, component, 4, checkpoint)
        assert_equal(before, resumed.environment.task.curriculum_manager.checkpoint_state_dict())
        checker = VerifyFinite(resumed)
        resumed.callbacks.append(checker)
        parameters = [p.detach().clone() for p in resumed.algorithm.policy.parameters()]
        resumed.train()
        delta = sum(float((p.detach() - old).square().sum())
                    for p, old in zip(resumed.algorithm.policy.parameters(), parameters)) ** .5
        assert checker.updates == 2 and resumed.current_iteration == 3 and delta > 0
        report.update(passed=True, initial_updates=2, resumed_updates=2,
                      resume_parameter_delta_l2=delta, finite_signals=True)
    finally:
        if resumed is not None:
            resumed.close()
        source.close()
        report["cleanup_returned"] = True
        (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMMAND CURRICULUM VERIFICATION PASSED", flush=True)


if __name__ == "__main__":
    main()
