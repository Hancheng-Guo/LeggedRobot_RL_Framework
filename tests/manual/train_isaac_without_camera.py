"""Run one real Isaac Sim PPO update without a camera or Fabric output.

Run from the repository root using the Python environment with Isaac Sim.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application_entry import ApplicationEntry  # noqa: E402
from envs.simulators.isaac_sim_runtime import IsaacSimRuntime  # noqa: E402
from envs.vector_env import VectorEnv  # noqa: E402
from rl.algorithms.ppo import PPO  # noqa: E402


def profile_training_calls():
    timings: dict[str, list[float]] = {}
    originals = []
    for owner, method_name in (
        (IsaacSimRuntime, "step"),
        (IsaacSimRuntime, "get_state"),
        (VectorEnv, "step"),
        (PPO, "act"),
        (PPO, "process_transition"),
        (PPO, "compute_returns"),
        (PPO, "update"),
    ):
        original = getattr(owner, method_name)
        label = f"{owner.__name__}.{method_name}"

        def measured(self, *args, operation=original, name=label, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = operation(self, *args, **kwargs)
            torch.cuda.synchronize()
            timings.setdefault(name, []).append(time.perf_counter() - start)
            return result

        setattr(owner, method_name, measured)
        originals.append((owner, method_name, original))
    return timings, originals


def print_timings(timings: dict[str, list[float]]) -> None:
    print("TRAINING TIMINGS (CUDA synchronized; nested rows overlap):", flush=True)
    for name, values in timings.items():
        print(
            f"  {name}: calls={len(values)}, total={sum(values):.3f} s, "
            f"mean={1000 * sum(values) / len(values):.2f} ms",
            flush=True,
        )


def main() -> None:
    start = time.perf_counter()
    timings, originals = profile_training_calls()
    try:
        with ApplicationEntry("unitree_go1_isaac_cuda_headless_train_test") as app:
            print("Starting camera-free PPO training.", flush=True)
            app.train()
            if app.stage_manager.continue_training:
                raise RuntimeError("PPO stage did not complete its first update.")
            print_timings(timings)
            print(
                f"CAMERA-FREE TRAINING PASSED in {time.perf_counter() - start:.2f} s; "
                f"logs: {app.save_dir / 'logs' / 'training.log'}",
                flush=True,
            )
    finally:
        for owner, method_name, original in reversed(originals):
            setattr(owner, method_name, original)


if __name__ == "__main__":
    main()
