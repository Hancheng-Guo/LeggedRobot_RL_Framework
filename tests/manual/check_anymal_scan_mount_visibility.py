"""Compare ANYmal C scan-mount occlusion against the original side placement.

Run with python -m tests.manual.check_anymal_scan_mount_visibility.
The rays target a base-parallel plane and stop at the horizontal floor.
"""
from itertools import product
from pathlib import Path

import mujoco
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "assets/anymal_c/MJCF/scene.xml"))
    data = mujoco.MjData(model)
    offsets = np.array([
        (0.12 * (q + r / 2), 0.12 * np.sqrt(3) * r / 2, 0)
        for q, r in product(range(-4, 5), repeat=2)
        if max(abs(q), abs(r), abs(q + r)) <= 4
    ])
    visual_meshes = np.zeros(6, dtype=np.uint8)
    visual_meshes[2] = 1
    for placement in ("side", "asset"):
        for hip_delta in (0, -0.24, 0.24):
            mujoco.mj_resetDataKeyframe(model, data, model.key("standing").id)
            for leg in ("RF", "LF", "RH", "LH"):
                data.qpos[model.joint(f"{leg}_HFE").qposadr] += hip_delta
            mujoco.mj_forward(model, data)
            base = data.body("base")
            blocked = 0
            blocked_centres = 0
            for leg in ("RF", "LF", "RH", "LH"):
                origin = data.site(f"scan_{leg.lower()}").xpos
                if placement == "side":
                    x = 0.21 if leg.endswith("F") else -0.21
                    y = -0.15 if leg.startswith("R") else 0.15
                    origin = base.xpos + base.xmat.reshape(3, 3) @ [x, y, -0.05]
                centre = data.geom(leg).xpos.copy()
                centre[2] = base.xpos[2] - 0.50
                for offset in offsets:
                    direction = centre + offset - origin
                    direction /= np.linalg.norm(direction)
                    floor_distance = -origin[2] / direction[2]
                    geom_id = np.empty(1, dtype=np.int32)
                    distance = mujoco.mj_ray(
                        model, data, origin, direction, visual_meshes, True, -1, geom_id,
                    )
                    if 0 <= distance < floor_distance:
                        blocked += 1
                        blocked_centres += int(not offset.any())
            print(f"{placement} HFE offset {hip_delta:+.2f} rad: "
                  f"{blocked}/244 rays blocked, including {blocked_centres}/4 centres")


if __name__ == "__main__":
    main()
