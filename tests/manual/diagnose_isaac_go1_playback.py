"""Run the Go1 playback check with startup phase markers and stack dumps."""

from __future__ import annotations

import faulthandler
import functools
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.simulators.isaac_sim_model import IsaacSimModelConverter
from envs.simulators.isaac_sim_runtime import IsaacSimRuntime
from tests.manual.verify_isaac_go1_camera_activation import main


def mark(message: str) -> None:
    print(f"DIAG {datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def trace_method(owner: Any, name: str) -> None:
    original = getattr(owner, name)

    @functools.wraps(original)
    def traced(*args: Any, **kwargs: Any) -> Any:
        mark(f"BEGIN {owner.__name__}.{name}")
        try:
            result = original(*args, **kwargs)
        except BaseException:
            mark(f"ERROR {owner.__name__}.{name}")
            raise
        mark(f"END {owner.__name__}.{name}")
        return result

    setattr(owner, name, traced)


for method_name in (
    "_build_scene",
    "_build_metadata",
    "_prepare_state_buffers",
    "_capture_default_state",
):
    trace_method(IsaacSimRuntime, method_name)
trace_method(IsaacSimModelConverter, "convert_if_needed")

original_start = IsaacSimRuntime._start_application


@functools.wraps(original_start)
def traced_start(self: IsaacSimRuntime) -> None:
    mark("BEGIN IsaacSimRuntime._start_application")
    original_start(self)
    mark("END IsaacSimRuntime._start_application")

    # Isaac modules are imported only after SimulationApp has started.
    from isaacsim.core.api import World
    from isaacsim.core.cloner import GridCloner

    for owner, method_name in (
        (GridCloner, "clone"),
        (GridCloner, "filter_collisions"),
        (World, "reset"),
    ):
        trace_method(owner, method_name)
    mark("Installed Kit phase markers")


IsaacSimRuntime._start_application = traced_start

if __name__ == "__main__":
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
