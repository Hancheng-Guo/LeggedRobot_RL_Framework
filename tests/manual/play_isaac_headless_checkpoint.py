"""Play an Isaac Sim checkpoint with an optional RGB camera."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application_entry import ApplicationEntry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-time", required=True, help="Timestamp in checkpoint directory name.")
    parser.add_argument(
        "--app-name",
        default="unitree_go1_isaac_cuda_test",
    )
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-plays", type=int, default=1)
    parser.add_argument("--formats", nargs="+", choices=("gif", "mp4"), default=["gif"])
    parser.add_argument(
        "--render-mode",
        choices=("none", "rgb_array"),
        default="rgb_array",
        help="Create an RGB camera and GIF, or run policy playback without images.",
    )
    args = parser.parse_args()
    if args.num_steps <= 0 or args.num_plays <= 0:
        parser.error("--num-steps and --num-plays must be positive")

    with ApplicationEntry(args.app_name, train_time=args.train_time) as app:
        # StageManager has not constructed the evaluation environment yet.
        # Use the archived training policy and task with one environment.
        app.stage_manager.component["environment"] = [
            {
                "type": "vector_env",
                "config_path": str(PROJECT_ROOT / "configs/environments/vector_env_1.yaml"),
            }
        ]
        if args.render_mode == "rgb_array":
            app.stage_manager.component["simulator"] = [
                {
                    "type": "isaac_sim",
                    "config_path": str(
                        PROJECT_ROOT / "configs/simulators/isaac_sim_unitree_go1_rgb_array.yaml"
                    ),
                }
            ]
        app.play(
            num_steps=args.num_steps,
            formats=args.formats,
            num_plays=args.num_plays,
        )
        if args.render_mode == "rgb_array":
            videos = sorted(
                path
                for format_name in args.formats
                for path in (app.save_dir / "videos").glob(f"*.{format_name}")
            )
            if not videos:
                raise RuntimeError("Playback did not produce a GIF.")
            print(f"CAMERA PLAYBACK PASSED: {videos[-1]}", flush=True)
        else:
            print("CAMERA-FREE PLAYBACK SKIPPED", flush=True)


if __name__ == "__main__":
    main()
