"""Compare the three scan modes on actual assets, with optional geometry playback.

python -m tests.manual.verify_scan_surfaces isaac_sim anymal_c --image scan.png
Use --play to follow moving rays in two orthographic views.
On Windows hosts with conflicting OpenMP runtimes, set MKL_THREADING_LAYER
to SEQUENTIAL for the Isaac verification process.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from utils import Component, ComponentInfo, RuntimeContext, load_yaml


def draw_rays(origins, directions, distance, ground, optical_axes, angle, title):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1000, 520), "#16242f")
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), title + "   cyan: ground   orange: self   white: centre   violet: field of view")
    hits = origins[:, None] + directions * distance[..., None]
    finite_hits = hits[np.isfinite(hits).all(axis=-1)]
    points = np.concatenate((origins - .12, origins + .12, finite_hits))
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    scale = min(440 / max(np.ptp(points, axis=0)[:2].max(), .01),
                400 / max(np.ptp(points[:, 2]), .01))
    for column, side in enumerate((0, 1)):
        def project(p):
            return (250 + 500 * column + scale * (p[side] - centre[side]), 280 - scale * (p[2] - centre[2]))
        draw.text((20+column*500, 44), ("XZ", "YZ")[column])
        for start, rays, lengths, is_ground, axis in zip(
            origins, directions, distance, ground, optical_axes, strict=True,
        ):
            # Project the actual cone rim into each orthographic view.
            tangent = np.cross(axis, [1, 0, 0] if abs(axis[0]) < .9 else [0, 1, 0])
            tangent /= np.linalg.norm(tangent)
            angles = np.linspace(0, 2*np.pi, 65)
            circle = np.cos(angles[:, None])*tangent + np.sin(angles[:, None])*np.cross(axis, tangent)
            rim = start + .12*(np.cos(angle)*axis + np.sin(angle)*circle)
            draw.line([project(p) for p in rim], fill="#b596e2")
            for i, (ray, length) in enumerate(zip(rays, lengths, strict=True)):
                if not np.isfinite(length):
                    continue
                hit = project(start + ray * length)
                draw.line([project(start), hit], fill="white" if i == 0 else "#496275")
                x, y = hit
                draw.ellipse((x-2,y-2,x+2,y+2), fill="#49dbdc" if is_ground[i] else "#ffa248")
            x,y = project(start)
            draw.ellipse((x-4,y-4,x+4,y+4), fill="#fff19c")
    return image


def set_test_pose(sim, isaac, roll_degrees, joint_offset):
    """Roll the whole robot and move one hip without advancing physics."""
    angle = np.deg2rad(roll_degrees) / 2
    quat = [np.cos(angle), np.sin(angle), 0., 0.]
    if isaac:
        runtime = sim._require_backend()
        positions, _ = runtime._articulation.get_world_poses()
        orientations = torch.tensor(quat, device=sim.context.device, dtype=sim.context.dtype)
        runtime._articulation.set_world_poses(positions=positions,
            orientations=runtime._warp(orientations.repeat(sim.num_envs, 1)))
        joints = runtime._tensor(runtime._articulation.get_dof_positions()).clone()
        joints[:, 1] += joint_offset
        runtime._articulation.set_dof_positions(runtime._warp(joints))
        runtime._update_articulation_kinematics()
    else:
        import mujoco
        for model, data in zip(sim.models, sim.datas, strict=True):
            data.qpos[3:7] = quat
            data.qpos[8] += joint_offset
            mujoco.mj_forward(model, data)


def verify_hits(sim, rays, state, mounts, height, angle_deg):
    """Independently reconstruct targets, Euclidean ranges and signed heights for env 0."""
    from itertools import product
    def host(tensor):
        return tensor.detach().cpu().numpy()
    bases = state.qpos[0,sim.model_context.base_pos_qpos_ids]
    quat = host(state.qpos[0,sim.model_context.base_quat_qpos_ids])
    def rotate(vectors):
        cross = 2 * np.cross(quat[1:], vectors)
        return vectors + quat[0] * cross + np.cross(quat[1:], cross)
    base = host(bases)
    mount_pos, mount_axis = sim.resolve_scan_mounts(tuple(mounts))
    origins = base + rotate(host(mount_pos))
    axes = rotate(host(mount_axis))
    np.testing.assert_allclose(host(rays["origins"][0]), origins, atol=2e-6)
    feet = host(state.geom_xpos[0,sim.model_context.foot_geom_ids])
    up = rotate(np.array([0.,0.,1.]))
    centres = feet - ((feet-base) @ up + height)[:,None]*up
    coords = sorted((max(abs(q),abs(r),abs(q+r)),r,q) for q,r in product(range(-4,5),repeat=2)
                    if max(abs(q),abs(r),abs(q+r)) <= 4)
    offsets = rotate(np.array([[.12*(q+r/2),.12*np.sqrt(3)*r/2,0.] for _,r,q in coords]))
    direction = centres[:,None]+offsets-origins[:,None]
    direction /= np.linalg.norm(direction,axis=-1,keepdims=True)
    emit = (direction*axes[:,None]).sum(-1) >= np.cos(np.deg2rad(angle_deg))-1e-7
    np.testing.assert_allclose(host(rays["directions"][0]),np.where(emit[...,None],direction,0.),atol=2e-6)
    with np.errstate(divide="ignore"):
        ground = -origins[:,None,2]/direction[...,2]
    ground = np.where(emit & (ground>=0) & (ground<=2),ground,np.nan)
    np.testing.assert_allclose(host(rays["ground"][0]),ground,rtol=2e-5,atol=2e-6)
    nearest = np.fmin(ground,host(rays["robot"][0]))
    hit_z = origins[:,None,2] + direction[...,2]*nearest
    expected = np.where(emit & np.isfinite(nearest),feet[:,None,2]-hit_z,0.)
    np.testing.assert_allclose(host(state.foot_terrain_scan_heights[0]),expected,atol=2e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("mujoco","isaac_sim"))
    parser.add_argument("robot", choices=("anymal_c","unitree_go1"))
    parser.add_argument("--device", help="Torch device (defaults to CUDA for Isaac and CPU for MuJoCo)")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    isaac = args.backend == "isaac_sim"
    if isaac:
        from envs.simulators.isaac_sim import IsaacSimSimulator as Simulator
    else:
        from envs.simulators.mujoco import MujocoSimulator as Simulator
    device = torch.device(args.device or ("cuda:0" if isaac else "cpu"))
    ctx = RuntimeContext(device, torch.float32, 1, 0, True, root, root)
    sim = Simulator(ctx)
    cfg = root / "configs/simulators" / f"{args.backend}_{args.robot}_{'headless' if isaac else 'rgb_array'}.yaml"
    params = load_yaml(cfg)
    params.pop("render_mode", None)
    component = Component(None,None,None,None,ComponentInfo(args.backend,cfg),None)
    mounts = ["scan_rf","scan_lf","scan_rh","scan_lh"]
    term = {"mount_names": mounts, "reference_height": .50 if args.robot=="anymal_c" else .27,
            "max_angle_deg": 80.}
    args_config = {"terms": {"foot_terrain_scan": term}}
    try:
        sim.config_update(component, num_envs=3, simulator_state_manager_config=args_config, **params)
        sim.reset()
        poses = {}
        rays = {}
        query = sim.query_surface_rays

        def capture(origins, directions, max_distance, env_ids, *, include_robot):
            distance, ground_hit = query(
                origins=origins, directions=directions, max_distance=max_distance,
                env_ids=env_ids, include_robot=include_robot,
            )
            rays.update(origins=origins, directions=directions)
            ground = sim.query_ground_rays(origins, directions, max_distance, env_ids)
            rays["ground"] = ground
            if include_robot:
                robot = sim.query_robot_rays(origins, directions, max_distance, env_ids)
                rays["robot"] = robot
                torch.testing.assert_close(distance, torch.fmin(ground, robot), equal_nan=True,
                                           atol=2e-5, rtol=1e-5)
                expected_ground = ground.isfinite() & (~robot.isfinite() | (ground < robot))
                torch.testing.assert_close(ground_hit, expected_ground)
            return distance, ground_hit

        # Compare the unified query with independent ground and robot queries in the same frame.
        with patch.object(sim, "query_surface_rays", capture):
            for mode in ("first_surface","ground_only","visible_ground"):
                term["hit_mode"] = mode
                sim._build_simulator_state_manager(args_config)
                state = sim.get_state()
                poses[mode] = state.clone()
                ids = torch.tensor([2,0], device=device)
                subset = sim.get_state(ids)
                torch.testing.assert_close(subset.foot_terrain_scan_heights,state.foot_terrain_scan_heights[ids])
                torch.testing.assert_close(subset.foot_terrain_scan_valid,state.foot_terrain_scan_valid[ids])
                assert sim.get_state(ids[:0]).foot_terrain_scan_valid.shape == (0,4,61)
            first, floor, visible = (poses[key] for key in ("first_surface","ground_only","visible_ground"))
            torch.testing.assert_close(first.foot_terrain_scan_heights, visible.foot_terrain_scan_heights)
            blocked = first.foot_terrain_scan_valid & ~visible.foot_terrain_scan_valid
            print("valid/ground/blocked",first.foot_terrain_scan_valid.sum().item(), visible.foot_terrain_scan_valid.sum().item(), blocked.sum().item(),flush=True)
            assert blocked.any() and visible.foot_terrain_scan_valid.any()
            feet = floor.geom_xpos[:, sim.model_context.foot_geom_ids, 2, None].expand(-1,-1,61)
            mask = floor.foot_terrain_scan_valid
            torch.testing.assert_close(floor.foot_terrain_scan_heights[mask], feet[mask], atol=2e-5,rtol=0)
            print(args.backend,args.robot,"self hits per foot",blocked[0].sum(-1).tolist(),flush=True)
            verify_hits(sim, rays, sim.get_state(), mounts, term["reference_height"], term["max_angle_deg"])
            origin, direction = rays["origins"], rays["directions"]
            clear = visible.foot_terrain_scan_valid
            # Sweep the hip in both directions: nominal poses differ across assets.
            for joint_offset in (.35, -.35, .65, -.65):
                sim.reset()
                set_test_pose(sim, isaac, 0., joint_offset)
                moved = sim.query_robot_rays(origin, direction, 2., None)
                newly_blocked = clear & moved.isfinite() & (moved < rays["ground"])
                if newly_blocked.any():
                    break
            assert newly_blocked.any(), "Expected the leg to enter at least one previously clear beam"
            sim.reset()
            restored = sim.query_robot_rays(origin, direction, 2., None)
            assert (~restored.isfinite() | (restored > rays["ground"]))[newly_blocked].all()
            set_test_pose(sim, isaac, 14., .22)
            tilted = sim.get_state()
            verify_hits(sim, rays, tilted, mounts, term["reference_height"], term["max_angle_deg"])
            assert not torch.equal(tilted.foot_terrain_scan_heights, visible.foot_terrain_scan_heights)
            sim.reset()
            if args.play or args.image:
                axis_local = sim.resolve_scan_mounts(tuple(mounts))[1]
                if args.play:
                    import tkinter as tk
                    from PIL import ImageTk
                    window = tk.Tk()
                    panel = tk.Label(window)
                    panel.pack()
                def frame():
                    state = sim.get_state()
                    quat = state.qpos[0,sim.model_context.base_quat_qpos_ids].cpu().numpy()
                    local = axis_local.cpu().numpy()
                    cross = 2 * np.cross(quat[1:], local)
                    axes = local + quat[0] * cross + np.cross(quat[1:], cross)
                    lengths = torch.fmin(rays["ground"], rays["robot"])
                    return draw_rays(*(v[0].cpu().numpy() for v in (
                        rays["origins"],rays["directions"],lengths,state.foot_terrain_scan_valid)),
                        axes,np.deg2rad(term["max_angle_deg"]),f"{args.robot} / {args.backend}")
                if args.image:
                    frame().save(args.image)
                if args.play:
                    def update():
                        sim.step(first.ctrl)
                        picture = ImageTk.PhotoImage(frame())
                        panel.configure(image=picture)
                        panel.image = picture
                        window.after(50,update)
                    update()
                    window.mainloop()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
