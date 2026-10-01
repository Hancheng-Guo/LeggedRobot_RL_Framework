"""Build and close IsaacSimSimulator without VectorEnv or Runner."""

from __future__ import annotations

import argparse
import faulthandler
import sys
import torch
import yaml
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulators.isaac_sim import IsaacSimSimulator
from utils import Component, RuntimeContext


def report(message: str) -> None:
    print(f"SIMULATOR SHUTDOWN: {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument(
        "--preallocate-cuda-tensor",
        action="store_true",
        help="Allocate the VectorEnv episode-step tensor before starting Isaac.",
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=ROOT / "configs/simulators/isaac_sim_unitree_go1_rgb_array.yaml",
    )
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")

    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
    config = yaml.safe_load(args.sim_config.read_text(encoding="utf-8"))
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    config["model_path"] = model_path.resolve()
    config["train_render_mode"] = None
    context = RuntimeContext(
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        num_threads=4,
        seed=0,
        deterministic_ops=False,
        load_dir=ROOT,
        save_dir=ROOT,
    )
    torch.set_num_threads(context.num_threads)
    episode_steps = None
    if args.preallocate_cuda_tensor:
        episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=context.device)
        report(f"preallocated CUDA episode-step tensor: {tuple(episode_steps.shape)}")
    component = Component(
        runner=None,
        algorithm=None,
        policy=None,
        environment=None,
        simulator=None,
        task=None,
    )
    simulator = IsaacSimSimulator(context=context)
    try:
        report("config_update begin")
        simulator.config_update(component=component, num_envs=args.num_envs, **config)
        report("config_update returned")
    finally:
        report("close begin")
        simulator.close()
        report("close returned")
        faulthandler.cancel_dump_traceback_later()
    report("script finished")


if __name__ == "__main__":
    main()
