"""Launch TensorBoard for a selected training run."""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from colorama import just_fix_windows_console

from utils import (
    LOG_FORMAT,
    ColoredFormatter,
    get_logger,
    launch_tensorboard,
    select_run_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_logdir(
    checkpoints_dir: Path,
    value: str = ""
) -> Path:
    
    requested = Path(value.strip()).expanduser() if value.strip() else None
    run_value = (
        str(requested.parent)
        if requested is not None and requested.name == "tensorboard"
        else value
    )
    logdir = select_run_dir(checkpoints_dir, run_value) / "tensorboard"

    if not logdir.is_dir():
        raise FileNotFoundError(f"TensorBoard log directory not found: {logdir}")
    return logdir.resolve()


def _serve(logdir: Path) -> None:
    root_logger = get_logger()
    root_logger.setLevel(logging.INFO)
    root_logger.propagate = False
    console_handler = logging.StreamHandler()
    if "NO_COLOR" not in os.environ and console_handler.stream.isatty():
        just_fix_windows_console()
        console_handler.setFormatter(ColoredFormatter(LOG_FORMAT))
    else:
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)
    try:
        url = launch_tensorboard(logdir)
        logger = get_logger(__name__)
        logger.info("TensorBoard dir: %s", logdir.as_posix())
        logger.info("TensorBoard url: %s", url)
        logger.info("Press Ctrl+C to stop TensorBoard.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        root_logger.removeHandler(console_handler)
        console_handler.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logdir",
        nargs="?",
        default="",
        help="Run directory name/path or TensorBoard directory; empty selects the latest dated run.",
    )
    parser.add_argument("--logdir", dest="logdir_option", help=argparse.SUPPRESS)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.serve:
        _serve(Path(args.logdir))
        return 0

    if args.logdir and args.logdir_option is not None:
        parser.error("Specify the run directory either positionally or with --logdir.")
    value = args.logdir_option if args.logdir_option is not None else args.logdir
    logdir = select_logdir(PROJECT_ROOT / "checkpoints", value)
    command = [
        sys.executable,
        "-m",
        "tools.launch_tensorboard",
        "--serve",
        str(logdir),
    ]
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
