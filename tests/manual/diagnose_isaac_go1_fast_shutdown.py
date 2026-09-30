"""Run the Go1 training/playback check with Kit fast shutdown enabled.

Diagnostic only: this overrides the SimulationApp constructor for this process.
"""

import faulthandler
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import isaacsim.simulation_app as simulation_app
from isaacsim.simulation_app import SimulationApp

original_simulation_app = SimulationApp


def fast_shutdown_application(config, *args, **kwargs):
    config = {**config, "fast_shutdown": True}
    print("DIAG forcing SimulationApp fast_shutdown=True", flush=True)
    return original_simulation_app(config, *args, **kwargs)


simulation_app.SimulationApp = fast_shutdown_application

from tests.manual.verify_isaac_go1_camera_activation import main


if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    main()
