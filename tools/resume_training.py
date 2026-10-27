"""Resume training from a historical run's checkpoint."""

import argparse
import os

from tools.play_checkpoint import PROJECT_ROOT, run_identity
from utils import select_run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", nargs="?", default="",
        help="Run directory name/path; empty selects the latest dated run.",
    )
    parser.add_argument(
        "--checkpoint", default="",
        help="Checkpoint path within the run; empty uses automatic selection.",
    )
    parser.add_argument(
        "--resume-mode", choices=("fork", "inplace"), default="fork",
        help="Write to a new run directory (fork) or the original run (inplace).",
    )
    args = parser.parse_args(argv)

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    run_dir = select_run_dir(checkpoints_dir, args.run_dir)
    if run_dir.parent != checkpoints_dir.resolve():
        parser.error(f"Run directory must be inside {checkpoints_dir}.")
    app_name, train_time = run_identity(run_dir)

    os.chdir(PROJECT_ROOT)
    from app import ApplicationEntry

    print(f"Resuming training: {run_dir} (mode: {args.resume_mode})", flush=True)
    application = ApplicationEntry(
        app_name, train_time=train_time,
        resume_mode=args.resume_mode,
        checkpoint=args.checkpoint.strip() or None,
    )
    try:
        application.train()
        application.save()
    finally:
        application.close()


if __name__ == "__main__":
    main()
