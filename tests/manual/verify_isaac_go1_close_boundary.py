"""Locate an Isaac exit crash relative to SimulationApp.close().

Each run starts a fresh process with the current Go1 test configuration. Logs
and a machine-readable summary are saved under temp/isaac_go1_close_boundary_*.
"""

from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def mark(message: str) -> None:
    print(f"MARK {datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def run_child(mode: str) -> int:
    faulthandler.enable(file=sys.stderr)
    # Keep Warp's generated kernel cache inside the project during this test.
    # The agent's filesystem sandbox cannot write to the usual NVIDIA cache.
    warp_cache = ROOT / "temp" / "isaac_warp_cache"
    warp_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WARP_CACHE_PATH", str(warp_cache))
    if os.name == "nt":
        # Suppress the Windows access-violation dialog, which otherwise makes
        # a crashed child look like a hung shutdown.
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    atexit.register(mark, "PYTHON_ATEXIT_REACHED")

    from isaacsim.simulation_app import SimulationApp
    from app.application_entry import ApplicationEntry
    from tests.manual.isaac_test_application import manual_isaac_application
    from tests.manual.verify_isaac_go1_camera_activation import main as run_full

    original_simulation_close = SimulationApp.close
    original_application_close = ApplicationEntry.close

    def trace_simulation_close(self, *args, **kwargs):
        mark("SIMULATION_APP_CLOSE_BEGIN")
        try:
            result = original_simulation_close(self, *args, **kwargs)
        except BaseException:
            mark("SIMULATION_APP_CLOSE_RAISED")
            raise
        mark("SIMULATION_APP_CLOSE_RETURNED")
        return result

    def trace_application_close(self, *args, **kwargs):
        mark("APPLICATION_ENTRY_CLOSE_BEGIN")
        try:
            result = original_application_close(self, *args, **kwargs)
        except BaseException:
            mark("APPLICATION_ENTRY_CLOSE_RAISED")
            raise
        mark("APPLICATION_ENTRY_CLOSE_RETURNED")
        return result

    mark(f"START pid={os.getpid()} mode={mode}")
    with patch.object(SimulationApp, "close", trace_simulation_close), patch.object(
        ApplicationEntry, "close", trace_application_close
    ):
        try:
            if mode == "full":
                run_full()
            else:
                with manual_isaac_application() as app:
                    app.train()
                    if app.stage_manager.continue_training:
                        raise RuntimeError("Go1 training stage did not complete")
                    mark("TRAINING_COMPLETED")
            mark("WORKFLOW_RETURNED")
        except BaseException:
            mark("WORKFLOW_RAISED")
            traceback.print_exc()
            return 23
    mark("CHILD_FINISHED")
    return 0


def run_parent(modes: list[str], repeats: int, timeout: int) -> None:
    output_dir = ROOT / "temp" / f"isaac_go1_close_boundary_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    print(f"Logs: {output_dir}", flush=True)

    for mode in modes:
        for repeat in range(1, repeats + 1):
            label = f"{mode}_{repeat}"
            stdout_path = output_dir / f"{label}.stdout.log"
            stderr_path = output_dir / f"{label}.stderr.log"
            print(f"Running {label} ...", flush=True)
            started = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "--child",
                        "--mode",
                        mode,
                    ],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                )
                timed_out = False
                try:
                    exit_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    exit_code = process.wait()

            markers = [
                line
                for line in stdout_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.startswith("MARK ")
            ]
            names = {line.split(" ", 3)[-1] for line in markers}
            if timed_out:
                boundary = "timeout"
            elif "SIMULATION_APP_CLOSE_BEGIN" not in names:
                boundary = "before_simulation_app_close"
            elif "SIMULATION_APP_CLOSE_RETURNED" not in names:
                boundary = "during_simulation_app_close"
            elif "APPLICATION_ENTRY_CLOSE_RETURNED" not in names:
                boundary = "after_simulation_app_close_during_application_close"
            elif "CHILD_FINISHED" not in names:
                boundary = "after_application_close_before_child_finish"
            elif exit_code != 0:
                boundary = "after_child_finish"
            else:
                boundary = "clean_exit"
            result = {
                "mode": mode,
                "repeat": repeat,
                "exit_code": exit_code,
                "exit_code_hex": f"0x{exit_code & 0xFFFFFFFF:08X}",
                "timed_out": timed_out,
                "boundary": boundary,
                "duration_seconds": round(time.monotonic() - started, 2),
                "markers": markers,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            results.append(result)
            (output_dir / "summary.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"{label}: exit={result['exit_code_hex']}, "
                f"boundary={boundary}, timeout={timed_out}",
                flush=True,
            )

    print(f"Summary: {output_dir / 'summary.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("train-only", "full"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--modes", nargs="+", choices=("train-only", "full"), default=["train-only", "full"]
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200, help="seconds per run")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("repeats and timeout must be positive")
    if args.child:
        if args.mode is None:
            parser.error("child requires --mode")
        return run_child(args.mode)
    run_parent(args.modes, args.repeats, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
