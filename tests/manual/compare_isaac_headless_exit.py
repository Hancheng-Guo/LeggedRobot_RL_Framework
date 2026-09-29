"""Compare graceful Isaac process exit with headless enabled and disabled.

Each case runs in a fresh child process. Results are written below temp/.
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
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def mark(message: str) -> None:
    print(f"MARK {datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def run_child(headless: bool, post_close_delay: int = 0) -> None:
    faulthandler.enable(file=sys.stderr)
    if os.name == "nt":
        # Avoid a native crash dialog holding the child open indefinitely.
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    atexit.register(mark, "PYTHON_ATEXIT_REACHED")

    from isaacsim.simulation_app import SimulationApp

    mark(f"START pid={os.getpid()} headless={headless} fast_shutdown=False")
    app = SimulationApp({"headless": headless, "fast_shutdown": False})
    mark("APP_STARTED")
    app.update()
    mark("UPDATE_DONE")
    mark("CLOSE_BEGIN")
    app.close(wait_for_replicator=False)
    mark("CLOSE_RETURNED")
    if post_close_delay:
        mark(f"POST_CLOSE_DELAY_BEGIN seconds={post_close_delay}")
        time.sleep(post_close_delay)
        mark("POST_CLOSE_DELAY_ENDED")
    mark("CHILD_FINISHED")


def run_parent(repeats: int, timeout: int) -> None:
    output_dir = ROOT / "temp" / f"isaac_headless_exit_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    print(f"Logs: {output_dir}", flush=True)

    for repeat in range(1, repeats + 1):
        # Reverse the order on alternate repeats to limit time/order bias.
        order = (True, False) if repeat % 2 else (False, True)
        for headless in order:
            label = f"headless_{int(headless)}_{repeat}"
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
                        "--headless" if headless else "--visible",
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
            names = {line.split(" ", 2)[-1] for line in markers}
            if timed_out:
                boundary = "timeout"
            elif "APP_STARTED" not in names:
                boundary = "startup_failure"
            elif "CLOSE_BEGIN" not in names:
                boundary = "before_close"
            elif "CLOSE_RETURNED" not in names:
                boundary = "during_close"
            elif exit_code != 0:
                boundary = "after_close"
            else:
                boundary = "clean_exit"
            result = {
                "headless": headless,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    child = parser.add_mutually_exclusive_group()
    child.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    child.add_argument("--visible", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=420, help="seconds per child")
    parser.add_argument("--post-close-delay", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1 or args.post_close_delay < 0:
        parser.error("repeats and timeout must be positive; delay must be nonnegative")
    if args.child:
        if not (args.headless or args.visible):
            parser.error("child requires --headless or --visible")
        run_child(args.headless, args.post_close_delay)
    else:
        run_parent(args.repeats, args.timeout)


if __name__ == "__main__":
    main()
