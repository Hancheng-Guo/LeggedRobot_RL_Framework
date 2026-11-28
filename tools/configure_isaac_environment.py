"""Configure a Windows Isaac Sim interpreter before native packages are imported."""

import sys
import sysconfig
from pathlib import Path


MKL_DEFAULT = 'os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")'


def main() -> None:
    if sys.platform != "win32":
        return

    startup_path = Path(sysconfig.get_path("purelib")) / "00_isaac_mkl_threading.pth"
    try:
        with startup_path.open("x", encoding="ascii") as startup:
            startup.write(f"import os; {MKL_DEFAULT}\n")
    except FileExistsError:
        if not any(
            line.startswith("import ") and MKL_DEFAULT in line.split("; ")
            for line in startup_path.read_text(encoding="ascii").splitlines()
        ):
            raise SystemExit(f"Existing Isaac startup configuration differs: {startup_path}")

    print(f"Configured Windows Isaac Sim startup: {startup_path}")


if __name__ == "__main__":
    main()
