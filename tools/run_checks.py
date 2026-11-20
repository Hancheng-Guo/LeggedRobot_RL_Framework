"""Run Pyright and Pytest in order, even if the first check fails."""

import argparse
import subprocess
import sys


CHECKS = {
    "mujoco": ("pyrightconfig.mujoco.json", ("-m", "core or mujoco", "tests")),
    "isaacsim": ("pyrightconfig.isaac.json", ("-m", "core or isaacsim", "tests")),
    "ros2": (
        "pyrightconfig.ros2.json",
        (
            "-k", "human_ros2 or ros2_keyboard",
            "tests/test_human_ros2_config.py",
            "tests/test_play_human_ros2.py",
            "tests/test_on_policy_runner.py",
        ),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=CHECKS)
    args = parser.parse_args()
    pyright_config, pytest_args = CHECKS[args.target]

    commands = (
        ("Pyright", [sys.executable, "-m", "pyright", "--project", pyright_config, "--pythonpath", sys.executable]),
        ("Pytest", [sys.executable, "-m", "pytest", *pytest_args]),
    )
    failed = False
    for name, command in commands:
        print(f"\nRunning {name}...", flush=True)
        result = subprocess.run(command, check=False)
        failed |= result.returncode != 0
        print(f"{name} exited with code {result.returncode}.", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
