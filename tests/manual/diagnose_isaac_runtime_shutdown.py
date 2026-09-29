"""Trace IsaacSimRuntime.close around the existing runtime-only profiler.

Pass profile_isaac_sim_runtime.py arguments unchanged. The final process exit
code remains important even when the completion marker is printed.
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

faulthandler.enable(file=sys.stderr)
faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)

if "--import-application-entry" in sys.argv:
    sys.argv.remove("--import-application-entry")
    from app.application_entry import ApplicationEntry  # noqa: F401

    print("RUNTIME SHUTDOWN: ApplicationEntry imported", flush=True)

from envs.simulators.isaac_sim_runtime import IsaacSimRuntime
from tests.manual.profile_isaac_sim_runtime import main


original_close = IsaacSimRuntime.close


def traced_close(self: IsaacSimRuntime) -> None:
    print("RUNTIME SHUTDOWN: close begin", flush=True)
    try:
        original_close(self)
    except BaseException:
        print("RUNTIME SHUTDOWN: close raised", flush=True)
        raise
    print("RUNTIME SHUTDOWN: close returned", flush=True)


IsaacSimRuntime.close = traced_close

if __name__ == "__main__":
    try:
        main()
        print("RUNTIME SHUTDOWN: script finished", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
