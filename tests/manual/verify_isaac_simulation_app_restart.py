"""Check whether SimulationApp can start twice in the same Python process."""

from __future__ import annotations

import argparse
import gc
import os
import time

from isaacsim.simulation_app import SimulationApp


def report(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    if args.cycles < 2:
        parser.error("--cycles must be at least 2")

    report(f"PID {os.getpid()} | starting {args.cycles} cycles")
    for cycle in range(1, args.cycles + 1):
        app = None
        report(f"CYCLE {cycle}: starting SimulationApp")
        try:
            app = SimulationApp({"headless": True, "fast_shutdown": False})
            report(f"CYCLE {cycle}: SimulationApp started")
            app.update()
            report(f"CYCLE {cycle}: update completed")
        finally:
            if app is not None:
                report(f"CYCLE {cycle}: closing SimulationApp")
                app.close(wait_for_replicator=False)
                report(f"CYCLE {cycle}: SimulationApp closed")
                app = None
            gc.collect()

    report("ISAAC SIMULATIONAPP SAME-PROCESS RESTART PASSED")


if __name__ == "__main__":
    main()
