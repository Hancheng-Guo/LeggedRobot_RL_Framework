"""Run an application by its configuration name."""

import argparse
from pathlib import Path

from app import ApplicationEntry


def main(app_name: str) -> None:
    config_path = Path("configs") / f"{app_name}.yaml"
    if not app_name or not config_path.is_file():
        raise SystemExit(f"Unknown app config: {app_name!r}")

    with ApplicationEntry(app_name) as app:
        app.train()
        app.save()
        app.play(num_steps=500, formats="gif")
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_name")
    main(parser.parse_args().app_name)
