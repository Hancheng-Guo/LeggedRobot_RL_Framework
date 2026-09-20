from __future__ import annotations

import argparse
import logging
import threading
from typing import Any

from isaacsim.simulation_app import SimulationApp


LOGGER = logging.getLogger("isaac_shutdown_diagnostic")


class KitLogBridge:
    """Small replica of the project's Kit-to-Python logging bridge."""

    def __init__(self) -> None:
        self._logging: Any = None
        self._handle: Any = None
        self._forwarding = threading.local()
        self._levels: dict[int, int] = {}

    def start(self) -> None:
        import carb.logging

        self._levels = {
            carb.logging.LEVEL_WARN: logging.WARNING,
            carb.logging.LEVEL_ERROR: logging.ERROR,
            carb.logging.LEVEL_FATAL: logging.CRITICAL,
        }
        self._logging = carb.logging.acquire_logging()
        self._handle = self._logging.add_logger(self._forward)
        LOGGER.info("Kit logging bridge registered.")

    def stop(self) -> None:
        if self._logging is None or self._handle is None:
            return
        self._logging.remove_logger(self._handle)
        self._handle = None
        self._logging = None
        LOGGER.info("Kit logging bridge removed.")

    def _forward(
        self,
        source: str,
        level: int,
        filename: str,
        line_number: int,
        message: str,
    ) -> None:
        project_level = self._levels.get(level)
        if project_level is None:
            return
        if getattr(self._forwarding, "active", False):
            return
        self._forwarding.active = True
        try:
            LOGGER.log(
                project_level,
                "[Isaac Sim: %s] %s",
                source,
                message.rstrip(),
            )
        finally:
            self._forwarding.active = False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolate Isaac Sim shutdown lifecycle failures."
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "bridge-before", "bridge-after", "world"),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    LOGGER.info("Starting diagnostic mode: %s", arguments.mode)

    bridge = KitLogBridge()
    if arguments.mode == "bridge-before":
        bridge.start()

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": False,
        }
    )
    LOGGER.info("SimulationApp initialized.")

    if arguments.mode in {"bridge-after", "world"}:
        bridge.start()

    world: Any = None
    if arguments.mode == "world":
        from isaacsim.core.api import World

        LOGGER.info("Creating World.")
        world = World(
            physics_dt=1.0 / 200.0,
            rendering_dt=1.0 / 60.0,
            backend="torch",
            device="cuda:0",
        )
        world.reset()
        world.step(render=False)
        LOGGER.info("World initialized and stepped.")

    if world is not None:
        world.stop()
        world.clear()
        LOGGER.info("World cleared.")

    bridge.stop()
    LOGGER.info("Closing SimulationApp.")
    app.close(wait_for_replicator=False)
    print("DIAGNOSTIC COMPLETED SUCCESSFULLY", flush=True)


if __name__ == "__main__":
    main()
