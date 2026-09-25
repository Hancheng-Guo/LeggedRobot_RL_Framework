import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.context import RuntimeContext
from envs.simulators.mujoco import MujocoSimulator
from utils.component import Component


MODEL_PATH = PROJECT_ROOT / "assets" / "unitree_go1" / "MJCF" / "scene.xml"


def median_us(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples) / 1_000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MuJoCo state fields.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    context = RuntimeContext(
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_threads=1,
        seed=0,
        deterministic_ops=True,
        load_dir=PROJECT_ROOT,
        save_dir=PROJECT_ROOT,
    )
    simulator = MujocoSimulator(context)
    simulator.config_update(
        component=Component(None, None, None, None, None, None),
        num_envs=args.num_envs,
        model_path=MODEL_PATH,
        sim_dt=0.002,
        frame_skip=10,
        step_workers=1,
        foot_geom_names=("FR", "FL", "RR", "RL"),
        floor_geom_names=("floor",),
        reset_keyframe="home",
    )
    simulator.reset()
    action = torch.zeros(args.num_envs, simulator.model_context.nu)
    for _ in range(args.warmup):
        simulator.step(action)
        simulator.get_state()

    datas = simulator.datas
    models = simulator.models
    contact_state = simulator._get_contact_state(models, datas)
    sections = {
        "basic_state": lambda: simulator._get_basic_state(datas),
        "base_velocity": lambda: simulator._get_base_velocity_state(datas),
        "geom_xvel": lambda: simulator._get_geom_xvel(models, datas),
        "contact_state": lambda: simulator._get_contact_state(models, datas),
        "foot_contact": lambda: simulator._get_foot_contact_state(*contact_state),
        "get_state_total": simulator.get_state,
    }

    try:
        print("section,median_us,share_of_total")
        timings = {
            name: median_us(function, args.repeats)
            for name, function in sections.items()
        }
        total = timings["get_state_total"]
        for name, elapsed_us in timings.items():
            print(f"{name},{elapsed_us:.2f},{elapsed_us / total:.3f}")
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
