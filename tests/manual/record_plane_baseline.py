"""Record stage-1 plane behavior and timings using the real vector environment."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from envs.vector_env import VectorEnv
from utils import Component, ComponentInfo, RuntimeContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["mujoco", "isaac_sim"])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180,
                        help="Total worker timeout, including Isaac startup and cleanup.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_envs < 2 or args.warmup < 0 or args.steps <= 0 or args.timeout <= 0:
        parser.error("Require num-envs >= 2, warmup >= 0, steps > 0")
    if not args.worker:
        # Keep a hung native shutdown bounded without calling os._exit or
        # confusing a measurements marker with successful process cleanup.
        started = time.time()
        worker = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                   *sys.argv[1:], "--worker"])
        timed_out = False
        try:
            exit_code = worker.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            worker.kill()
            exit_code = worker.wait()
        status = dict(exit_code=exit_code, timed_out=timed_out,
                      timeout_seconds=args.timeout,
                      measurements_written=args.output.exists() and args.output.stat().st_mtime >= started,
                      elapsed_seconds=time.time() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".process.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"BASELINE PROCESS: {status}", flush=True)
        raise SystemExit(124 if timed_out else exit_code)
    device = torch.device("cpu" if args.backend == "mujoco" else "cuda:0")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    context = RuntimeContext(device=device, dtype=torch.float32, num_threads=4,
        seed=0, deterministic_ops=False, load_dir=ROOT, save_dir=ROOT)
    config = "mujoco_unitree_go1_rgb_array" if args.backend == "mujoco" else "isaac_sim_unitree_go1_headless"
    component = Component(None, None, None, None,
        ComponentInfo(args.backend, ROOT / f"configs/simulators/{config}.yaml"),
        ComponentInfo("locomotion", ROOT / "configs/tasks/locomotion_unitree_go1_standing.yaml"))
    env = VectorEnv(context)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
    def timed(operation):
        synchronize()
        start = time.perf_counter()
        result = operation()
        synchronize()
        return result, (time.perf_counter() - start) * 1000
    try:
        _, initialization = timed(lambda: env.config_update(component, num_envs=args.num_envs, max_episode_steps=1000))
        obs, reset_ms = timed(env.reset)
        sim = env.simulator
        initial = sim.get_state().clone()
        action = torch.zeros(args.num_envs, sim.model_context.nu, device=device)
        times = []
        contacts = []
        for index in range(args.warmup + args.steps):
            result, ms = timed(lambda: env.step(action))
            assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]).all()
            if index >= args.warmup:
                times.append(ms)
                contacts.append(sim.get_state().foot_ground_contact.sum(dim=0).cpu().tolist())
        before = sim.get_state().clone()
        ids = torch.tensor([0], device=device)
        _, partial_reset_ms = timed(lambda: sim.reset(ids))
        after = sim.get_state().clone()
        isolation = {name: torch.equal(value[1:], after.as_dict()[name][1:])
                     for name, value in before.as_dict().items()}
        reset_repeatable = torch.allclose(after.qpos[0], initial.qpos[0], atol=1e-6, rtol=0)
        points = after.geom_xpos[:, sim.model_context.foot_geom_ids, :2]
        heights = sim.get_ground_heights(points)
        assert all(isolation.values()), isolation
        assert reset_repeatable
        assert torch.equal(heights, torch.zeros_like(heights))
        report = dict(backend=args.backend, measured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            python=sys.version, platform=platform.platform(),
            torch=torch.__version__, device=str(device), seed=0, num_envs=args.num_envs,
            warmup=args.warmup, samples=args.steps, sim_dt=sim.sim_dt, frame_skip=sim.frame_skip,
            simulator_config=config, task_config="locomotion_unitree_go1_standing",
            initialization_ms=initialization, reset_ms=reset_ms, partial_reset_ms=partial_reset_ms,
            observation_shape=list(obs.shape), state_shapes={k:list(v.shape) for k,v in initial.as_dict().items()},
            initial_foot_contacts=initial.foot_ground_contact.cpu().tolist(),
            sampled_contact_counts_per_foot=contacts,
            final_foot_normal_forces=before.foot_contact_normal_force.cpu().tolist(),
            partial_reset_isolation=isolation, reset_qpos_repeatable=bool(reset_repeatable),
            plane_height_zero=True, env_step_mean_ms=statistics.mean(times),
            env_step_median_ms=statistics.median(times), env_step_p95_ms=sorted(times)[(95*len(times)+99)//100-1],
            env_steps_per_second=args.num_envs*1000/statistics.mean(times),
            timing_scope="VectorEnv.step includes physics, state, task signals, automatic resets; excludes PPO and rendering")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"BASELINE MEASUREMENTS PASSED: {args.output}", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        print("BASELINE CLEANUP BEGIN", flush=True)
        env.close()
        print("BASELINE CLEANUP RETURNED", flush=True)


if __name__ == "__main__":
    main()
