"""Compare Isaac Sim shutdown paths and whether errors survive process exit.

Run from the project root with the Isaac Python interpreter. Each case starts a
fresh child process; artifacts are written under temp/isaac_shutdown_exit_*.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASES = (
    "graceful-success",
    "graceful-explicit-exit-success",
    "graceful-explicit-exit-error",
    "fast-success",
    "fast-error-default",
    "fast-error-preserved",
)
ERROR_EXIT_CODE = 23


def mark(message: str) -> None:
    print(f"MARK {datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def run_child(case: str) -> None:
    faulthandler.enable()
    from isaacsim.simulation_app import SimulationApp

    fast = case.startswith("fast-")
    mark(f"START pid={os.getpid()} case={case} fast_shutdown={fast}")
    app = SimulationApp({"headless": True, "fast_shutdown": fast})
    mark("APP_STARTED")
    app.update()
    mark("UPDATE_DONE")

    error = case.startswith("fast-error-") or case == "graceful-explicit-exit-error"
    if error:
        try:
            raise RuntimeError("Intentional diagnostic failure")
        except RuntimeError:
            mark("INTENTIONAL_ERROR_RAISED")
            traceback.print_exc()

    exit_code = ERROR_EXIT_CODE if case == "fast-error-preserved" else 0
    mark(f"CLOSE_BEGIN requested_exit_code={exit_code}")
    app.close(wait_for_replicator=False, exit_code=exit_code)
    mark("CLOSE_RETURNED")
    if case.startswith("graceful-explicit-exit-"):
        final_code = ERROR_EXIT_CODE if error else 0
        mark(f"EXPLICIT_EXIT requested_exit_code={final_code}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(final_code)
    if error:
        raise RuntimeError("Intentional diagnostic failure after close")
    mark("CHILD_FINISHED")


def run_parent(cases: list[str], repeats: int, timeout: int) -> int:
    output_dir = ROOT / "temp" / f"isaac_shutdown_exit_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    print(f"Logs: {output_dir}", flush=True)

    for case in cases:
        for repeat in range(1, repeats + 1):
            label = f"{case}_{repeat}"
            stdout_path = output_dir / f"{label}.stdout.log"
            stderr_path = output_dir / f"{label}.stderr.log"
            print(f"Running {label} ...", flush=True)
            started = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--child", case],
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=stderr,
                )
                timed_out = False
                try:
                    return_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    return_code = process.wait()

            output = stdout_path.read_text(encoding="utf-8", errors="replace")
            markers = [line for line in output.splitlines() if line.startswith("MARK ")]
            result = {
                "case": case,
                "repeat": repeat,
                "pid": process.pid,
                "exit_code": return_code,
                "exit_code_hex": f"0x{return_code & 0xFFFFFFFF:08X}",
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
                f"{label}: exit={return_code} ({result['exit_code_hex']}), "
                f"timeout={timed_out}, last={last_marker}",
                flush=True,
            )

    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=360, help="seconds per child")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("--repeats and --timeout must be positive")
    if args.child:
        run_child(args.child)
        return 0
    return run_parent(args.cases, args.repeats, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
