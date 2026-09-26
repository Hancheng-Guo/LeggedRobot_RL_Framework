"""Profile the project's Isaac Sim runtime without starting PPO training.

Run from the repository root with the Python environment that has Isaac Sim.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.context import RuntimeContext  # noqa: E402
from envs.simulators.isaac_sim_backend import IsaacSimResetState  # noqa: E402
from envs.simulators.isaac_sim_runtime import IsaacSimRuntime  # noqa: E402


LOGGER = logging.getLogger("profile_isaac_sim_runtime")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--with-camera",
        action="store_true",
        help="Build the RGB camera and time render() separately.",
    )
    parser.add_argument(
        "--disable-fabric-output",
        action="store_true",
        help="Disable unused Fabric transform and velocity publishing for headless GPU tensor profiling.",
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=PROJECT_ROOT / "configs/simulators/isaac_sim_unitree_go1_rgb_array.yaml",
    )
    args = parser.parse_args()
    if args.num_envs <= 0 or args.warmup < 0 or args.steps <= 0:
        parser.error("--num-envs and --steps must be positive; --warmup must be nonnegative")
    if args.with_camera and args.disable_fabric_output:
        parser.error("--disable-fabric-output requires a run without --with-camera")
    return args


def elapsed_ms(operation) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def report(name: str, values: list[float]) -> None:
    ordered = sorted(values)
    p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
    LOGGER.info(
        "%-14s mean=%9.2f ms  median=%9.2f ms  p95=%9.2f ms",
        name,
        statistics.mean(values),
        statistics.median(values),
        p95,
    )


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config_path = args.sim_config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    context = RuntimeContext(
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        num_threads=4,
        seed=0,
        deterministic_ops=False,
        load_dir=PROJECT_ROOT,
        save_dir=PROJECT_ROOT,
    )
    torch.set_num_threads(context.num_threads)
    runtime = IsaacSimRuntime(context)
    reset_config = config.get("reset_state") or {}
    reset_state = IsaacSimResetState(
        base_position=tuple(reset_config["base_position"]) if "base_position" in reset_config else None,
        base_orientation=tuple(reset_config["base_orientation"]) if "base_orientation" in reset_config else None,
        joint_positions=reset_config.get("joint_positions", {}),
    )
    LOGGER.info(
        "Starting Isaac Sim benchmark: num_envs=%d, frame_skip=%d, camera=%s, disable_fabric_output=%s",
        args.num_envs,
        config["frame_skip"],
        args.with_camera,
        args.disable_fabric_output,
    )
    for method_name, label in (
        ("_start_application", "Isaac Sim application startup"),
        ("_build_scene", "scene construction and physics reset"),
        ("_build_metadata", "model metadata"),
        ("_prepare_state_buffers", "state buffers"),
        ("_capture_default_state", "default state capture"),
    ):
        original = getattr(runtime, method_name)

        def timed_phase(*, operation=original, name=label, phase=method_name):
            LOGGER.info("Starting %s.", name)
            start = time.perf_counter()
            operation()
            if phase == "_start_application" and args.disable_fabric_output:
                import carb.settings

                settings = carb.settings.get_settings()
                settings.set_bool("/physics/fabricUpdateTransformations", False)
                settings.set_bool("/physics/fabricUpdateVelocities", False)
                LOGGER.info(
                    "Fabric output disabled: transformations=%s, velocities=%s",
                    settings.get("/physics/fabricUpdateTransformations"),
                    settings.get("/physics/fabricUpdateVelocities"),
                )
            LOGGER.info("Completed %s in %.2f s.", name, time.perf_counter() - start)

        setattr(runtime, method_name, timed_phase)
    try:
        start = time.perf_counter()
        runtime.configure(
            num_envs=args.num_envs,
            model_path=model_path.resolve(),
            ros_package_paths=tuple(config.get("ros_package_paths") or ()),
            sim_dt=config["sim_dt"],
            frame_skip=config["frame_skip"],
            render_mode="rgb_array" if args.with_camera else None,
            env_spacing=config.get("env_spacing", 2.0),
            robot_prim_path=config.get("robot_prim_path", "/Robot"),
            base_body_prim_path=config.get("base_body_prim_path"),
            foot_body_prim_paths=tuple(config.get("foot_body_prim_paths") or ()),
            floor_prim_paths=tuple(config.get("floor_prim_paths") or ()),
            foot_contact_force_threshold=config.get("foot_contact_force_threshold", 15.0),
            camera_prim_path=config.get("camera_prim_path"),
            camera_resolution=tuple(config.get("camera_resolution", (640, 480))),
            merge_fixed_joints=config.get("merge_fixed_joints", False),
            allow_self_collision=config.get("allow_self_collision", False),
            joint_stiffness=config.get("joint_stiffness"),
            joint_damping=config.get("joint_damping"),
            reset_state=reset_state,
        )
        LOGGER.info("Runtime initialization: %.2f s", time.perf_counter() - start)
        physics_context = runtime._world.get_physics_context()
        LOGGER.info(
            "PhysX configuration:\n"
            "  simulation device: %s\n"
            "  GPU simulation: %s\n"
            "  GPU pipeline: %s\n"
            "  GPU dynamics: %s\n"
            "  broadphase: %s",
            runtime._simulation_manager.get_physics_sim_device(),
            physics_context.use_gpu_sim,
            physics_context.use_gpu_pipeline,
            physics_context.is_gpu_dynamics_enabled(),
            physics_context.get_broadphase_type(),
        )
        action = runtime.metadata.actuator_default_ctrl.repeat(args.num_envs, 1)
        step_times: list[float] = []
        state_times: list[float] = []
        render_times: list[float] = []
        for index in range(args.warmup + args.steps):
            step_ms = elapsed_ms(lambda: runtime.step(action, config["frame_skip"]))
            state_ms = elapsed_ms(runtime.get_state)
            render_ms = elapsed_ms(runtime.render) if args.with_camera else None
            if index >= args.warmup:
                step_times.append(step_ms)
                state_times.append(state_ms)
                if render_ms is not None:
                    render_times.append(render_ms)
            LOGGER.info(
                "%s %d/%d: step=%.2f ms, state=%.2f ms%s",
                "warmup" if index < args.warmup else "sample",
                index + 1 if index < args.warmup else index - args.warmup + 1,
                args.warmup if index < args.warmup else args.steps,
                step_ms,
                state_ms,
                "" if render_ms is None else f", render={render_ms:.2f} ms",
            )
        LOGGER.info("Summary (%d environments, %d physics substeps/control step):", args.num_envs, config["frame_skip"])
        report("step", step_times)
        report("get_state", state_times)
        if render_times:
            report("render", render_times)
        LOGGER.info("Approximate simulator throughput: %.0f env steps/s", args.num_envs * 1000 / (statistics.mean(step_times) + statistics.mean(state_times) + (statistics.mean(render_times) if render_times else 0)))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
