"""Play a historical checkpoint with keyboard commands carried by ROS 2."""

import argparse
import multiprocessing
import os
from pathlib import Path

from tools.play_checkpoint import PROJECT_ROOT, run_identity
from utils import select_run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_dir", nargs="?", default="",
        help="Run directory name/path; empty selects the latest dated run.",
    )
    parser.add_argument(
        "--playback-config", required=True,
        help="Playback config name under configs/playback, or a YAML path.",
    )
    args = parser.parse_args(argv)
    if not args.playback_config.strip():
        parser.error("Playback config name or path is required.")
    playback_config = Path(args.playback_config).expanduser()
    if not playback_config.is_absolute():
        if playback_config.parent == Path("."):
            playback_config = Path("configs/playback") / playback_config
        playback_config = PROJECT_ROOT / playback_config
    if not playback_config.suffix:
        playback_config = playback_config.with_suffix(".yaml")
    playback_config = playback_config.resolve()
    if not playback_config.is_file():
        parser.error(f"Playback config must be an existing file: {playback_config}")
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    run_dir = select_run_dir(checkpoints_dir, args.checkpoint_dir)
    if run_dir.parent != checkpoints_dir.resolve():
        parser.error(f"Run directory must be inside {checkpoints_dir}.")
    app_name, train_time = run_identity(run_dir)

    os.chdir(PROJECT_ROOT)
    from app import ApplicationEntry

    application = ApplicationEntry(app_name, train_time)
    try:
        application.play(
            mode="human_ros2", num_steps=None, num_plays=1,
            playback_config=playback_config,
        )
    except KeyboardInterrupt:
        pass
    finally:
        application.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
