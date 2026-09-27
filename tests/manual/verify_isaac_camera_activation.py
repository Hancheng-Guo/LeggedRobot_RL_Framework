"""Activate an RGB camera after camera-free physics in one SimulationApp."""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
from isaacsim.simulation_app import SimulationApp


def report(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--render-attempts", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 0 or args.render_attempts <= 0:
        parser.error("--steps and --render-attempts must be positive")

    report(f"PID {os.getpid()} | starting headless SimulationApp")
    app = SimulationApp(
        {
            "headless": True,
            "disable_viewport_updates": True,
            "fast_shutdown": False,
        }
    )
    world = None
    camera = None
    try:
        import carb.settings
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera

        settings = carb.settings.get_settings()
        settings.set_bool("/physics/fabricUpdateTransformations", False)
        settings.set_bool("/physics/fabricUpdateVelocities", False)
        report("PHASE 1: creating World without a camera or Fabric output")
        world = World(physics_dt=0.002, rendering_dt=0.002)
        world.scene.add_default_ground_plane()
        world.reset()
        for _ in range(args.steps):
            world.step(render=False)
        report(f"PHASE 1: completed {args.steps} camera-free physics steps")

        settings.set_bool("/physics/fabricUpdateTransformations", True)
        settings.set_bool("/physics/fabricUpdateVelocities", True)
        report("PHASE 2: creating RGB camera in the existing World")
        camera = Camera(
            prim_path="/World/Camera",
            position=np.asarray((2.5, 2.5, 1.8)),
            resolution=(320, 240),
        )
        camera.initialize()
        report("PHASE 2: camera initialized; requesting rendered frames")
        for attempt in range(1, args.render_attempts + 1):
            world.render()
            frame = camera.get_rgba()
            if frame is None:
                continue
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            frame = np.asarray(frame)
            if frame.ndim == 3 and frame.shape[:2] == (240, 320) and frame.shape[2] >= 3:
                report(f"PHASE 2: received RGB frame {frame.shape} on attempt {attempt}")
                report("ISAAC CAMERA ACTIVATION PASSED")
                return
        raise RuntimeError(
            f"No valid camera frame after {args.render_attempts} render attempts."
        )
    finally:
        if camera is not None:
            report("CLEANUP: destroying camera")
            camera.destroy()
            app.update()
            camera = None
            gc.collect()
        if world is not None:
            report("CLEANUP: stopping World")
            world.stop()
            world.clear()
        report("CLEANUP: closing SimulationApp")
        app.close(wait_for_replicator=False)
        report("CLEANUP: SimulationApp closed")


if __name__ == "__main__":
    main()
