import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "unitree_go1" / "MJCF" / "scene.xml"


def step(model, data, frame_skip: int) -> None:
    mujoco.mj_step(model, data, nstep=frame_skip)


def benchmark(
    model_path: Path,
    num_envs: int,
    frame_skip: int,
    workers: int,
    warmup_steps: int,
    measured_steps: int,
    repeats: int,
) -> tuple[float, float]:
    models = [mujoco.MjModel.from_xml_path(str(model_path)) for _ in range(num_envs)]
    datas = [mujoco.MjData(model) for model in models]
    executor = (
        None
        if workers == 1
        else ThreadPoolExecutor(max_workers=min(workers, num_envs))
    )

    def run_once() -> None:
        if executor is None:
            for model, data in zip(models, datas):
                step(model, data, frame_skip)
        else:
            tuple(executor.map(
                step,
                models,
                datas,
                (frame_skip for _ in range(num_envs)),
            ))

    try:
        for _ in range(warmup_steps):
            run_once()

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(measured_steps):
                run_once()
            samples.append(time.perf_counter() - start)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    median_seconds = statistics.median(samples)
    control_rate = measured_steps / median_seconds
    return control_rate, num_envs * control_rate


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark persistent MuJoCo step thread pools.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-envs", type=parse_int_list, default=[16, 32, 64])
    parser.add_argument("--workers", type=parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print("num_envs,workers,control_steps_per_second,env_steps_per_second,speedup")
    for num_envs in args.num_envs:
        baseline = None
        for workers in args.workers:
            control_rate, env_rate = benchmark(
                model_path=args.model,
                num_envs=num_envs,
                frame_skip=args.frame_skip,
                workers=workers,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                repeats=args.repeats,
            )
            if baseline is None:
                baseline = env_rate
            print(
                f"{num_envs},{workers},{control_rate:.2f},"
                f"{env_rate:.2f},{env_rate / baseline:.3f}"
            )


if __name__ == "__main__":
    main()
