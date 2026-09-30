"""Exercise Go1 playback with graceful Kit cleanup and explicit process exit.

Diagnostic only: the explicit exit happens after the existing manual Go1 test
returns, so ApplicationEntry and SimulationApp have already closed.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import traceback
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

faulthandler.enable(file=sys.stderr)
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
print("SHUTDOWN DIAG: importing project modules", flush=True)
from app.application_entry import ApplicationEntry
from envs.vector_env import VectorEnv
from tests.manual.isaac_test_application import manual_isaac_application
from tests.manual.verify_isaac_go1_camera_activation import main as run_go1_check
print("SHUTDOWN DIAG: project modules imported", flush=True)


def report(message: str) -> None:
    print(f"SHUTDOWN DIAG: {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--inject-playback-error",
        action="store_true",
        help="Raise a deliberate error on the first play call after training.",
    )
    group.add_argument(
        "--train-only",
        action="store_true",
        help="Train without a camera or playback, then inspect shutdown timing.",
    )
    group.add_argument(
        "--stop-before-runner-train",
        action="store_true",
        help="Build Runner, environment and PPO, then fail before training starts.",
    )
    group.add_argument(
        "--skip-runner-train",
        action="store_true",
        help="Build Runner, environment and PPO, then return without training.",
    )
    group.add_argument(
        "--skip-algorithm-build",
        action="store_true",
        help="Build Runner and environment without PPO, then return.",
    )
    group.add_argument(
        "--build-environment-only",
        action="store_true",
        help="Build only VectorEnv and simulator, without callbacks or PPO.",
    )
    group.add_argument(
        "--build-simulator-only",
        action="store_true",
        help="Build VectorEnv and simulator without task, callbacks or PPO.",
    )
    parser.add_argument(
        "--without-tensorboard",
        action="store_true",
        help="With --train-only, omit the TensorBoard callback for isolation.",
    )
    parser.add_argument(
        "--algorithm-first",
        action="store_true",
        help="With --train-only, close PPO before the Isaac environment.",
    )
    args = parser.parse_args()
    if (args.without_tensorboard or args.algorithm_first) and not args.train_only:
        parser.error("--without-tensorboard and --algorithm-first require --train-only")

    report("diagnostic imports complete")
    exit_code = 0
    try:
        callback_patch = nullcontext()
        order_patch = nullcontext()
        if (
            args.without_tensorboard
            or args.algorithm_first
            or args.stop_before_runner_train
            or args.skip_runner_train
            or args.skip_algorithm_build
            or args.build_environment_only
            or args.build_simulator_only
        ):
            from runners.on_policy import OnPolicyRunner

            report("runner imported for cleanup isolation")
        if args.without_tensorboard:
            original_build_callbacks = OnPolicyRunner._build_callbacks

            def build_without_tensorboard(self, callbacks):
                configured = self._callback_configs if callbacks is None else callbacks
                filtered = [
                    entry
                    for entry in configured or ()
                    if entry != "tensorboard"
                    and not (isinstance(entry, Mapping) and "tensorboard" in entry)
                ]
                report("omitting TensorBoard callback")
                return original_build_callbacks(self, filtered)

            callback_patch = patch.object(
                OnPolicyRunner, "_build_callbacks", build_without_tensorboard
            )
        if args.algorithm_first:

            def close_algorithm_first(self):
                self._run_callbacks("_on_close")
                if hasattr(self, "algorithm"):
                    self.algorithm.close()
                    del self.algorithm
                    report("algorithm closed before environment")
                if hasattr(self, "environment"):
                    self.environment.close()
                    del self.environment

            order_patch = patch.object(OnPolicyRunner, "close", close_algorithm_first)

        if args.train_only:
            with callback_patch, order_patch, manual_isaac_application() as app:
                report("training without playback")
                app.train()
                if app.stage_manager.continue_training:
                    raise RuntimeError("Go1 training stage did not complete")
                runtime = app.stage_manager.runner.environment.simulator._require_backend()
                if runtime._camera is not None:
                    raise AssertionError("Training unexpectedly created a camera")
                report("training completed without a camera")
        elif args.stop_before_runner_train:

            def stop_before_train(self):
                report("Runner built; intentionally stopping before train")
                raise RuntimeError("Intentional stop before Runner.train")

            with patch.object(OnPolicyRunner, "train", stop_before_train):
                with manual_isaac_application() as app:
                    app.train()
        elif (
            args.skip_runner_train
            or args.skip_algorithm_build
            or args.build_environment_only
            or args.build_simulator_only
        ):

            def skip_runner_train(self):
                report("Runner built; returning without training")

            algorithm_patch = nullcontext()
            if args.skip_algorithm_build or args.build_environment_only or args.build_simulator_only:

                def skip_algorithm_build(self, component):
                    report("omitting PPO algorithm construction")

                algorithm_patch = patch.object(
                    OnPolicyRunner, "_build_algorithm", skip_algorithm_build
                )

            callbacks_patch = nullcontext()
            stage_patch = nullcontext()
            if args.build_environment_only or args.build_simulator_only:

                def skip_callbacks(self, callbacks):
                    report("omitting Runner callbacks")

                def skip_stage_callback(self, transition):
                    report("omitting stage callback")

                callbacks_patch = patch.object(
                    OnPolicyRunner, "_build_callbacks", skip_callbacks
                )
                stage_patch = patch.object(
                    OnPolicyRunner, "_set_stage_transition", skip_stage_callback
                )

            task_patch = nullcontext()
            if args.build_simulator_only:

                def skip_task_build(self, component, model_context):
                    report("omitting task construction")

                task_patch = patch.object(VectorEnv, "_build_task", skip_task_build)

            with algorithm_patch, callbacks_patch, stage_patch, task_patch, patch.object(
                OnPolicyRunner, "train", skip_runner_train
            ):
                with manual_isaac_application() as app:
                    app.train()
                    report("training workflow returned without PPO steps")
        elif args.inject_playback_error:
            with patch.object(
                ApplicationEntry,
                "play",
                side_effect=RuntimeError("Intentional playback diagnostic failure"),
            ):
                run_go1_check()
        else:
            run_go1_check()
        report("Go1 check returned after application cleanup")
    except BaseException:
        exit_code = 23
        report("Go1 check failed")
        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        report(f"explicit process exit code={exit_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
