"""Measure sliding friction for two environments on one Isaac collision plane."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.simulators.isaac_sim_friction import IsaacSimFriction
from envs.simulators.isaac_sim_runtime import IsaacSimRuntime
from utils import RuntimeContext


def measure_sliding(world, sliders):
    sliders.set_world_poses(positions=np.array([[0., -1., .1], [0., 1., .1]], dtype=np.float32))
    sliders.set_velocities(
        linear_velocities=np.array([[4., 0., 0.]] * 2, dtype=np.float32),
        angular_velocities=np.zeros((2, 3), dtype=np.float32),
    )
    for _ in range(48):
        world.step(render=False)
    speeds = sliders.get_velocities()[0].numpy()[:, 0]
    return (4. - speeds) / (9.81 * 48 / 240)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor-friction", type=float, default=.5, choices=(0., .5))
    parser.add_argument("--floor-combine", default="average", choices=("average", "multiply"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    context = RuntimeContext(torch.device("cuda:0"), torch.float32, 4, 0, False, ROOT, ROOT / "temp")
    runtime = IsaacSimRuntime(context)
    try:
        runtime._start_application()
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        runtime._simulation_manager = SimulationManager
        world = World(physics_dt=1 / 240, rendering_dt=1 / 240, backend="torch", device="cuda:0")
        runtime._world = world
        floor = world.scene.add_default_ground_plane(
            static_friction=args.floor_friction, dynamic_friction=args.floor_friction,
        )
        paths = []
        for env_id, side in enumerate((-1., 1.)):
            path = f"/World/Slider_{env_id}"
            cube = UsdGeom.Cube.Define(world.stage, path)
            cube.CreateSizeAttr(.2)
            cube.AddTranslateOp().Set(Gf.Vec3d(0., side, .1))
            prim = cube.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.)
            paths.append(path)
        runtime._body_view = sliders = RigidPrim(paths)
        floor_path = runtime._find_floor_collision_prim_path(world.stage, floor.prim_path, Usd, UsdPhysics)
        floor_material, _ = UsdShade.MaterialBindingAPI(
            world.stage.GetPrimAtPath(floor_path)
        ).ComputeBoundMaterial("physics")
        PhysxSchema.PhysxMaterialAPI.Apply(floor_material.GetPrim()).CreateFrictionCombineModeAttr(
            args.floor_combine
        )
        runtime._friction = friction = IsaacSimFriction(
            world.stage, [floor_path], [[path] for path in paths],
        )
        world.reset()
        baseline = measure_sliding(world, sliders)
        if args.floor_combine == "multiply":
            baseline_target = .5 * args.floor_friction
        else:
            baseline_target = (.5 + args.floor_friction) / 2
        np.testing.assert_allclose(baseline, [baseline_target] * 2, atol=.045, rtol=0)
        friction.apply([0, 1], [.2, 1.4])
        distinct = measure_sliding(world, sliders)
        snapshot = friction.capture([0])
        friction.apply([0], [1.])
        updated = measure_sliding(world, sliders)
        friction.restore(snapshot)
        restored = measure_sliding(world, sliders)
        print("Measured friction", baseline, distinct, updated, restored, flush=True)
        np.testing.assert_allclose(distinct, baseline * [.2, 1.4], atol=.045, rtol=0)
        np.testing.assert_allclose(updated, baseline * [1., 1.4], atol=.045, rtol=0)
        np.testing.assert_allclose(restored, distinct, atol=.045, rtol=0)
        del friction, sliders, floor
    finally:
        runtime.close()
    print("ISAAC FRICTION SLIDING PASSED", flush=True)


if __name__ == "__main__":
    main()
