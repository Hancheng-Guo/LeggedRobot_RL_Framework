"""Play a selected historical training run."""

import argparse
import os
import re
from pathlib import Path

from utils import select_run_dir


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = re.compile(r"(.+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def run_identity(run_dir: Path) -> tuple[str, str]:
    match = RUN_NAME.fullmatch(run_dir.name)
    if match is None:
        raise ValueError(
            f"Run directory name must end with YYYY-MM-DD_HH-MM-SS: {run_dir.name}"
        )
    return match.group(1), match.group(2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default="",
        help="Run directory name/path; empty selects the latest dated run.",
    )
    args = parser.parse_args(argv)

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    run_dir = select_run_dir(checkpoints_dir, args.checkpoint_dir)
    if run_dir.parent != checkpoints_dir.resolve():
        parser.error(f"Run directory must be inside {checkpoints_dir}.")
    app_name, train_time = run_identity(run_dir)

    # ApplicationEntry resolves historical runs relative to the working directory.
    os.chdir(PROJECT_ROOT)
    from app import ApplicationEntry

    video_dir = run_dir / "videos"
    existing_videos = set(video_dir.glob("*"))
    print(f"Playing checkpoint: {run_dir}", flush=True)
    application = ApplicationEntry(app_name, train_time)
    try:
        application.play()
        new_videos = sorted(
            path for path in video_dir.glob("*")
            if path.is_file() and path not in existing_videos
        )
        if new_videos:
            for path in new_videos:
                print(f"Saved playback: {path}", flush=True)
        else:
            print("Playback finished without new video files.", flush=True)
    finally:
        application.close()


if __name__ == "__main__":
    main()
