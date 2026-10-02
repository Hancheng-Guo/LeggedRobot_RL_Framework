"""Run Pyright and Pytest in order, even if the first check fails."""

import argparse
import subprocess
import sys


CHECKS = {
    "mujoco": ("pyrightconfig.mujoco.json", "core or mujoco"),
    "isaacsim": ("pyrightconfig.isaac.json", "core or isaacsim"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=CHECKS)
    args = parser.parse_args()
    pyright_config, pytest_marker = CHECKS[args.target]

    commands = (
        ("Pyright", [sys.executable, "-m", "pyright", "--project", pyright_config, "--pythonpath", sys.executable]),
        ("Pytest", [sys.executable, "-m", "pytest", "-m", pytest_marker, "tests"]),
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
