"""Compare Isaac shutdown with and without SimulationApp's cleanup steps.

Run with the quadruped_isaac interpreter. Each case starts a fresh child process.
Results and full logs are written below temp/isaac_skip_cleanup_*.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def mark(message: str) -> None:
    print(f"MARK {datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def run_child(updates: int, skip_cleanup: bool) -> None:
    faulthandler.enable()
    if os.name == "nt":
        # Keep an access-violation dialog from blocking the parent indefinitely.
        # Both sides of the comparison use the same error-mode setting.
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

    from isaacsim.simulation_app import SimulationApp

    mark(
        f"START pid={os.getpid()} updates={updates} "
        f"skip_cleanup={skip_cleanup} headless=True fast_shutdown=False"
    )
    app = SimulationApp({"headless": True, "fast_shutdown": False})
    mark("APP_STARTED")
    for index in range(1, updates + 1):
        app.update()
        mark(f"UPDATE_{index}_DONE")
    mark("CLOSE_BEGIN")
    app.close(wait_for_replicator=False, skip_cleanup=skip_cleanup)
    mark("CLOSE_RETURNED")
    mark("CHILD_FINISHED")


def run_parent(update_counts: list[int], repeats: int, timeout: int) -> None:
    output_dir = ROOT / "temp" / f"isaac_skip_cleanup_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    print(f"Logs: {output_dir}", flush=True)

    for updates in update_counts:
        for skip_cleanup in (False, True):
            for repeat in range(1, repeats + 1):
                label = f"updates_{updates}_skip_{int(skip_cleanup)}_{repeat}"
                stdout_path = output_dir / f"{label}.stdout.log"
                stderr_path = output_dir / f"{label}.stderr.log"
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--child",
                    "--updates",
                    str(updates),
                ]
                if skip_cleanup:
                    command.append("--skip-cleanup")
                print(f"Running {label} ...", flush=True)
                started = time.monotonic()
                with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                    "w", encoding="utf-8"
                ) as stderr:
                    process = subprocess.Popen(
                        command, cwd=ROOT, stdout=stdout, stderr=stderr
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
                result = {
                    "updates": updates,
                    "skip_cleanup": skip_cleanup,
                    "repeat": repeat,
                    "exit_code": exit_code,
                    "exit_code_hex": f"0x{exit_code & 0xFFFFFFFF:08X}",
                    "timed_out": timed_out,
                    "duration_seconds": round(time.monotonic() - started, 2),
                    "markers": markers,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                }
                results.append(result)
                (output_dir / "summary.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                last_marker = markers[-1] if markers else "NO_MARKER"
                print(
                    f"{label}: exit={result['exit_code_hex']}, "
                    f"timeout={timed_out}, last={last_marker}",
                    flush=True,
                )

    print(f"Summary: {output_dir / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--updates", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--skip-cleanup", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=420, help="seconds per case")
    args = parser.parse_args()
    if any(count < 0 for count in args.updates) or args.repeats < 1 or args.timeout < 1:
        parser.error("updates must be nonnegative; repeats and timeout must be positive")
    if args.child:
        if len(args.updates) != 1:
            parser.error("child requires exactly one update count")
        run_child(args.updates[0], args.skip_cleanup)
    else:
        run_parent(args.updates, args.repeats, args.timeout)


if __name__ == "__main__":
    main()
