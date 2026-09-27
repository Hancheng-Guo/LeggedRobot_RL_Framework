"""Train Go1 without a camera, then play with one in the same Isaac app."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application_entry import ApplicationEntry  # noqa: E402


def report(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    video_dir: Path
    previous: set[Path]
    with ApplicationEntry("unitree_go1_isaac_cuda_headless_train_test") as app:
        report("PHASE 1: training Go1 without a camera")
        app.train()
        if app.stage_manager.continue_training:
            raise RuntimeError("Go1 training stage did not complete.")

        environment = app.stage_manager.runner.environment
        simulator = environment.simulator
        runtime = simulator._require_backend()
        if runtime.render_mode is not None or runtime._camera is not None:
            raise AssertionError("Training unexpectedly created a camera.")
        application_id = id(runtime._app)
        report(f"PHASE 1: training completed; SimulationApp id={application_id}")

        import carb.settings
        from isaacsim.sensors.camera import Camera

        settings = carb.settings.get_settings()
        settings.set_bool("/physics/fabricUpdateTransformations", True)
        settings.set_bool("/physics/fabricUpdateVelocities", True)
        report("PHASE 2: enabling Fabric output and creating a Go1 camera")
        runtime._build_camera(Camera)
        runtime._camera.initialize()
        runtime._camera_env_index = int(
            runtime._default_root_positions[:, :2]
            .square()
            .sum(dim=-1)
            .argmin()
            .item()
        )
        runtime.render_mode = "rgb_array"
        simulator.render_mode = "rgb_array"

        video_dir = app.save_dir / "videos"
        previous = set(video_dir.glob("*.gif"))
        report("PHASE 3: playing through ApplicationEntry.play()")
        app.play(num_steps=5, formats="gif", num_plays=1)
        if id(runtime._app) != application_id:
            raise AssertionError("Playback replaced SimulationApp.")
        if app.stage_manager.runner.environment is not environment:
            raise AssertionError("Playback replaced the Go1 environment.")
        created = set(video_dir.glob("*.gif")) - previous
        if not created:
            raise AssertionError("Go1 playback produced no GIF.")
        report(f"PHASE 3: created {sorted(created)[-1]}")

    report("ISAAC GO1 SAME-APP CAMERA PLAYBACK PASSED")


if __name__ == "__main__":
    main()
