"""Train Go1 without a camera, then play with one in the same Isaac app."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.manual.isaac_test_application import manual_isaac_application  # noqa: E402


def report(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    video_dir: Path
    previous: set[Path]
    with manual_isaac_application() as app:
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

        video_dir = app.save_dir / "videos"
        previous = set(video_dir.glob("*.gif"))
        report("PHASE 2: activating camera through ApplicationEntry.play()")
        app.play(num_steps=5, formats="gif", num_plays=1)
        if runtime.render_mode is not None or runtime._camera is not None:
            raise AssertionError("Playback did not restore the camera-free runtime.")
        import carb.settings

        settings = carb.settings.get_settings()
        if settings.get("/physics/fabricUpdateTransformations") or settings.get(
            "/physics/fabricUpdateVelocities"
        ):
            raise AssertionError("Playback did not restore Fabric output settings.")
        if id(runtime._app) != application_id:
            raise AssertionError("Playback replaced SimulationApp.")
        if app.stage_manager.runner.environment is not environment:
            raise AssertionError("Playback replaced the Go1 environment.")
        created = set(video_dir.glob("*.gif")) - previous
        if not created:
            raise AssertionError("Go1 playback produced no GIF.")
        report(f"PHASE 2: created {sorted(created)[-1]}")

        report("PHASE 3: repeating playback after cleanup")
        app.play(num_steps=5, formats="gif", num_plays=1)
        if runtime.render_mode is not None or runtime._camera is not None:
            raise AssertionError("Second playback did not restore the runtime.")
        if len(set(video_dir.glob("*.gif")) - previous) < 2:
            raise AssertionError("Second playback produced no GIF.")

    report("ISAAC GO1 SAME-APP CAMERA PLAYBACK PASSED")


if __name__ == "__main__":
    main()
