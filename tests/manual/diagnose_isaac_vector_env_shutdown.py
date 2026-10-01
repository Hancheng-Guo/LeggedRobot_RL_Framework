"""Build and close VectorEnv with Isaac, without ApplicationEntry or Runner."""

from __future__ import annotations

import argparse
import faulthandler
import sys
import torch
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import (
    create_runtime_context,
    resolve_device,
    set_seed,
)
from envs.vector_env import VectorEnv
from utils import Component, ComponentInfo, RuntimeContext


def report(message: str) -> None:
    print(f"VECTOR ENV SHUTDOWN: {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--defer-cuda-buffer",
        action="store_true",
        help="Create VectorEnv.current_episode_steps on CUDA after Isaac startup.",
    )
    parser.add_argument(
        "--use-runtime-context-factory",
        action="store_true",
        help="Build context through the same seed/device setup as ApplicationEntry.",
    )
    parser.add_argument(
        "--indexed-runtime-device",
        action="store_true",
        help="Use cuda:0 when constructing the context through the factory.",
    )
    parser.add_argument(
        "--fixed-runtime-seed",
        action="store_true",
        help="Use seed 0 when constructing the context through the factory.",
    )
    parser.add_argument(
        "--initialize-seed",
        action="store_true",
        help="Apply the entry point's seed initialization before Isaac startup.",
    )
    parser.add_argument(
        "--probe-device",
        action="store_true",
        help="Apply the entry point's CUDA availability check before Isaac startup.",
    )
    parser.add_argument(
        "--unindexed-cuda",
        action="store_true",
        help="Use torch.device('cuda') as the application entry point does.",
    )
    parser.add_argument(
        "--threads-before-seed",
        action="store_true",
        help="Set PyTorch threads before seeding, matching create_runtime_context.",
    )
    parser.add_argument(
        "--trace-close-steps",
        action="store_true",
        help="Mark each Isaac world/app cleanup operation during close.",
    )
    parser.add_argument(
        "--skip-native-release",
        action="store_true",
        help="Diagnostic only: bypass the simulation-manager native interface release.",
    )
    args = parser.parse_args()
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
    if args.use_runtime_context_factory:
        context = create_runtime_context(
            runtime_config={
                "device": "cuda:0" if args.indexed_runtime_device else "cuda",
                "dtype": "float32",
                "num_threads": 4,
                "seed": 0 if args.fixed_runtime_seed else None,
                "deterministic_ops": False,
            },
            load_dir=ROOT,
            save_dir=ROOT,
        )
        report("context built through ApplicationEntry factory")
    else:
        if args.probe_device:
            resolve_device("cuda")
            report("CUDA availability checked")
        if args.threads_before_seed:
            torch.set_num_threads(4)
            report("PyTorch thread count set before seeding")
        if args.initialize_seed:
            set_seed(seed=0, deterministic_ops=False)
            report("random seeds initialized")
        context = RuntimeContext(
            device=torch.device("cuda" if args.unindexed_cuda else "cuda:0"),
            dtype=torch.float32,
            num_threads=4,
            seed=0,
            deterministic_ops=False,
            load_dir=ROOT,
            save_dir=ROOT,
        )
        if not args.threads_before_seed:
            torch.set_num_threads(context.num_threads)
    component = Component(
        runner=None,
        algorithm=None,
        policy=None,
        environment=None,
        simulator=ComponentInfo(
            type="isaac_sim",
            config=ROOT / "configs/simulators/isaac_sim_unitree_go1_rgb_array.yaml",
        ),
        task=None,
    )
    environment = VectorEnv(context=context)

    def skip_task(self, component, model_context):
        report("task construction omitted")

    original_zeros = torch.zeros
    deferred = False

    def defer_episode_steps(*shape, **kwargs):
        nonlocal deferred
        if (
            not deferred
            and shape == (128,)
            and kwargs.get("dtype") == torch.long
            and kwargs.get("device") == context.device
        ):
            deferred = True
            torch.zeros = original_zeros
            report("CUDA episode-step allocation deferred until after Isaac startup")
            return original_zeros(*shape, **{**kwargs, "device": torch.device("cpu")})
        return original_zeros(*shape, **kwargs)

    try:
        zeros_patch = patch.object(torch, "zeros", defer_episode_steps) if args.defer_cuda_buffer else patch.object(torch, "zeros", original_zeros)
        with patch.object(VectorEnv, "_build_task", skip_task), zeros_patch:
            report("config_update begin")
            environment.config_update(
                component=component,
                num_envs=128,
                max_episode_steps=1000,
            )
            report("config_update returned")
        if args.defer_cuda_buffer:
            if not deferred:
                raise AssertionError("Episode-step allocation was not intercepted")
            environment.current_episode_steps = original_zeros(
                environment.num_envs,
                dtype=torch.long,
                device=context.device,
            )
            report("CUDA episode-step tensor allocated after Isaac startup")
    finally:
        report("close begin")
        with ExitStack() as patches:
            if args.trace_close_steps:
                runtime = environment.simulator._backend
                if runtime is None:
                    raise AssertionError("Isaac runtime was not configured")

                def trace_method(target, name, label):
                    if isinstance(target, type):
                        original = getattr(target, name)

                        def traced_classmethod(cls, *method_args, **method_kwargs):
                            report(f"{label} begin")
                            try:
                                return original(*method_args, **method_kwargs)
                            finally:
                                report(f"{label} ended")

                        patches.enter_context(
                            patch.object(target, name, classmethod(traced_classmethod))
                        )
                        return
                    original = getattr(type(target), name)

                    def traced(instance, *method_args, **method_kwargs):
                        report(f"{label} begin")
                        try:
                            return original(instance, *method_args, **method_kwargs)
                        finally:
                            report(f"{label} ended")

                    patches.enter_context(patch.object(type(target), name, traced))

                if runtime._world is not None:
                    trace_method(runtime._world, "stop", "World.stop")
                    trace_method(runtime._world, "clear", "World.clear")
                if runtime._simulation_manager is not None:
                    trace_method(
                        runtime._simulation_manager,
                        "invalidate_physics",
                        "SimulationManager.invalidate_physics",
                    )
                    trace_method(runtime._simulation_manager, "_shutdown", "SimulationManager._shutdown")
                    trace_method(runtime._simulation_manager, "_reset", "SimulationManager._reset")
                    from isaacsim.core.simulation_manager.impl import extension as simulation_manager_extension

                    native_module = simulation_manager_extension._simulation_manager
                    original_release = native_module.release_simulation_manager_interface

                    def traced_release(*release_args, **release_kwargs):
                        report("native simulation-manager release begin")
                        try:
                            if args.skip_native_release:
                                report("native simulation-manager release skipped")
                                return None
                            return original_release(*release_args, **release_kwargs)
                        finally:
                            report("native simulation-manager release ended")

                    patches.enter_context(
                        patch.object(
                            native_module,
                            "release_simulation_manager_interface",
                            traced_release,
                        )
                    )
                trace_method(runtime, "_stop_log_bridge", "log bridge stop")
                if runtime._app is not None:
                    trace_method(runtime._app, "close", "SimulationApp.close")
                    import carb

                    original_log_info = carb.log_info

                    def trace_app_log(message, *log_args, **log_kwargs):
                        if str(message).startswith("SimulationApp.close:"):
                            report(f"Kit close log: {message}")
                        return original_log_info(message, *log_args, **log_kwargs)

                    patches.enter_context(patch.object(carb, "log_info", trace_app_log))
            environment.close()
        report("close returned")
        faulthandler.cancel_dump_traceback_later()
    report("script finished")


if __name__ == "__main__":
    main()
