"""Run one real Isaac Sim PPO update without a camera or Fabric output.

Run from the repository root using the Python environment with Isaac Sim.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application_entry import ApplicationEntry  # noqa: E402
from envs.simulators.isaac_sim_runtime import IsaacSimRuntime  # noqa: E402


def main() -> None:
    original_start_application = IsaacSimRuntime._start_application

    def start_without_fabric_output(self: IsaacSimRuntime) -> None:
        if self.render_mode is not None:
            raise RuntimeError("This training diagnostic requires render_mode=None.")
        original_start_application(self)

        import carb.settings

        settings = carb.settings.get_settings()
        settings.set_bool("/physics/fabricUpdateTransformations", False)
        settings.set_bool("/physics/fabricUpdateVelocities", False)
        print(
            "Fabric output: transformations="
            f"{settings.get('/physics/fabricUpdateTransformations')}, "
            "velocities="
            f"{settings.get('/physics/fabricUpdateVelocities')}",
            flush=True,
        )

    IsaacSimRuntime._start_application = start_without_fabric_output
    try:
        start = time.perf_counter()
        with ApplicationEntry("unitree_go1_isaac_cuda_headless_train_test") as app:
            print("Starting camera-free PPO training.", flush=True)
            app.train()
            if app.stage_manager.continue_training:
                raise RuntimeError("PPO stage did not complete its first update.")
            print(
                f"CAMERA-FREE TRAINING PASSED in {time.perf_counter() - start:.2f} s; "
                f"logs: {app.save_dir / 'logs' / 'training.log'}",
                flush=True,
            )
    finally:
        IsaacSimRuntime._start_application = original_start_application


if __name__ == "__main__":
    main()
