"""Stage 4 native scan acceptance. Run each backend and phase in a fresh process."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from verify_command_curriculum import VerifyFinite, configure
from utils import Component, ComponentInfo, RuntimeContext, load_yaml


def verify_geometry(env, isaac, output, play_scan):
    from unittest.mock import patch

    sim = env.simulator
    state = sim.get_state()
    indices = torch.tensor([2, 0], device=state.qpos.device)
    partial = sim.get_state(indices)
    torch.testing.assert_close(partial.foot_terrain_scan_heights, state.foot_terrain_scan_heights[indices])
    # A failed reset after terrain/friction assignment must restore the measured surface.
    with patch.object(env.task, "reset", side_effect=RuntimeError("injected reset failure")):
        try:
            env.reset()
        except RuntimeError as error:
            assert str(error) == "injected reset failure"
        else:
            raise AssertionError("Expected injected failure")
    restored = sim.get_state()
    torch.testing.assert_close(restored.foot_terrain_scan_heights, state.foot_terrain_scan_heights)
    torch.testing.assert_close(restored.foot_terrain_scan_valid, state.foot_terrain_scan_valid)

    lower, upper = sim.get_ground_bounds(indices)
    centre = (lower + upper) / 2
    # Sample triangle interiors, edges, tile boundaries and the outer skirt.
    offsets = centre.new_tensor([[1.23, .37], [1.27, .33], [1.2, .4], [2., .4], [2.2, .4]])
    xy = centre[:, None] + offsets
    if isaac:
        terrain = sim._require_backend()._terrain
        border = terrain._map_upper.to(centre) + terrain.border_width
    else:
        border = centre.new_tensor(sim._terrain.origin[:2] + sim._terrain.outer_half_size)
    edges = torch.stack((border - .15, border + .25)).expand(len(indices), -1, -1)
    xy = torch.cat((xy, edges), dim=1)
    z = sim.get_ground_heights(xy, indices)
    expected = z.isfinite()
    points = torch.cat((xy, z.nan_to_num()[..., None]), dim=-1)
    direction = torch.nn.functional.normalize(centre.new_tensor([-.04, .03, -1.]), dim=0)
    origins = points - direction * .65
    rays = direction.expand(*points.shape[:-1], 1, 3).contiguous()
    distances, ground = sim.query_surface_rays(origins, rays, 1., indices, include_robot=False)
    torch.testing.assert_close(ground.squeeze(-1), expected)
    torch.testing.assert_close(distances.squeeze(-1)[expected], torch.full_like(z[expected], .65),
                               atol=5e-5, rtol=0.)
    data = {"geometry_rays": expected.numel()}
    if isaac:
        from omni.physx import get_physx_scene_query_interface
        physx = get_physx_scene_query_interface()
        # Identical batch for the GPU and PhysX timing; prepare host inputs outside timing.
        repeats = 61
        rays = rays.expand(-1, -1, repeats, -1).contiguous()
        starts = origins.cpu().tolist()
        vector = direction.cpu().tolist()
        sim.query_ground_rays(origins, rays, 1., indices)
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(8):
            sim.query_ground_rays(origins, rays, 1., indices)
        torch.cuda.synchronize()
        gpu = (perf_counter() - start) / 8
        start = perf_counter()
        for row, row_valid in zip(starts, expected.tolist()):
            for origin, valid in zip(row, row_valid):
                for _ in range(repeats):
                    hit = physx.raycast_closest(tuple(origin), tuple(vector), 1.)
                assert hit["hit"] == valid
                if valid:
                    assert abs(hit["distance"] - .65) < 5e-5 and hit["collision"] == terrain.prim_path
        physics = perf_counter() - start
        assert gpu < physics
        data.update(gpu_batch_ms=1000*gpu, physx_batch_ms=1000*physics, batch_rays=rays.numel()//3)
    env.current_episode_steps[0] = env.max_episode_steps - 1
    action = state.ctrl.new_zeros((env.num_envs, env.action_dim))
    next_obs, terminal_obs, _, _, truncated, info = env.step(action)
    assert truncated[0] and not truncated[1:].any()
    slices = env.task.observation_manager.term_slices
    key = slices["foot_terrain_scan_heights"]
    torch.testing.assert_close(terminal_obs[:, key], info["state/foot_terrain_scan_heights"].flatten(1).clamp(-1., .6))
    torch.testing.assert_close(next_obs[:, key], sim.get_state().foot_terrain_scan_heights.flatten(1).clamp(-1., .6))
    render_scan(env, output, play_scan)
    return data


def render_scan(env, output, play):
    from unittest.mock import patch
    import numpy as np
    from verify_scan_surfaces import draw_rays
    sim = env.simulator
    query = sim.query_surface_rays
    rays = {}

    def capture(*, origins, directions, max_distance, env_ids, include_robot):
        distance, ground = query(origins, directions, max_distance, env_ids, include_robot=include_robot)
        rays.update(origins=origins, directions=directions, distance=distance, ground=ground)
        return distance, ground

    _, local_axes = sim.resolve_scan_mounts(("scan_rf", "scan_lf", "scan_rh", "scan_lh"))
    def frame():
        state = sim.get_state()
        q = state.qpos[0, sim.model_context.base_quat_qpos_ids].cpu().numpy()
        local = local_axes.cpu().numpy()
        cross = 2 * np.cross(q[1:], local)
        axes = local + q[0] * cross + np.cross(q[1:], cross)
        return draw_rays(*(v[0].cpu().numpy() for v in rays.values()), axes, np.pi/2, "Terrain scan")

    with patch.object(sim, "query_surface_rays", capture):
        frame().save(output / "scan_geometry.png")
        if play:
            import tkinter as tk
            from PIL import ImageTk
            window = tk.Tk()
            panel = tk.Label(window)
            panel.pack()
            action = local_axes.new_zeros((env.num_envs, env.action_dim))
            def advance():
                env.step(action)
                image = ImageTk.PhotoImage(frame())
                panel.configure(image=image)
                panel.image = image
                window.after(50, advance)
            advance()
            window.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["mujoco", "isaac_sim"])
    parser.add_argument("phase", choices=["source", "resume", "baseline", "benchmark", "plane", "slope", "geometry"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--play-scan", action="store_true", help="Display live rays in the geometry phase")
    args = parser.parse_args()
    output = args.output.resolve() / args.backend
    output.mkdir(parents=True, exist_ok=True)
    isaac = args.backend == "isaac_sim"
    go1 = args.phase in ("plane", "slope")
    robot = "unitree_go1" if go1 else "anymal_c"
    suffix = "headless" if isaac else "rgb_array"
    sim = load_yaml(ROOT / f"configs/simulators/{args.backend}_{robot}_{suffix}.yaml")
    sim["render_mode"] = "rgb_array"
    task_name = "unitree_go1_standing" if go1 else "anymal_c_phase"
    task = load_yaml(ROOT / f"configs/tasks/locomotion_{task_name}.yaml")
    if go1:
        value = {} if args.phase == "plane" else {"max_slope": .3}
        task.setdefault("randomization_manager_config", {}).setdefault("terms", {})["terrain"] = {
            "category": {"type": "Fixed", "params": {"values": {args.phase: value}}},
            "patch_size": [4, 4], "resolution": .1, "spawn_size": [1.5, 1.2],
        }
    if args.phase == "geometry":
        task["randomization_manager_config"]["terms"]["friction"] = {
            "type": "Uniform", "params": {"factors": [.5, 1.2]}}
    if args.phase != "baseline":
        sim["simulator_state_manager_config"] = {"terms": {"foot_terrain_scan": {
            "mount_names": ["scan_rf", "scan_lf", "scan_rh", "scan_lh"],
            "reference_height": .27 if go1 else .5, "hit_mode": "visible_ground",
        }}}
        task["observation_manager_config"]["terms"].update(
            foot_terrain_scan_heights={"clip_range": [-1., .6]}, foot_terrain_scan_valid={})
    def config(name, data):
        path = output / (name + ".yaml")
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path
    component = Component(None,
        ComponentInfo("ppo", ROOT / "configs/algorithms/ppo_num_batch_4.yaml"),
        ComponentInfo("actor_critic", ROOT / "configs/policies/actor_critic_direct_actuation.yaml"),
        ComponentInfo("vector_env", config("env", {"num_envs": 4, "max_episode_steps": 32})),
        ComponentInfo(args.backend, config("sim", sim)), ComponentInfo("locomotion", config("task", task)))
    torch.manual_seed(5)
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if isaac else "cpu")
    ctx = RuntimeContext(device, torch.float32, 4, 5, False, ROOT, output)
    checkpoint = output / "scan.pt"
    runner = configure(ctx, component, 2 if args.phase == "resume" else 1,
                       checkpoint if args.phase == "resume" else None)
    record = {}
    try:
        env = runner.environment
        obs = env.reset()
        if args.phase == "resume":
            state = env.simulator.get_state()
            expected = torch.load(output / "reset_scan.pt", map_location=device, weights_only=True)
            torch.testing.assert_close(state.foot_terrain_scan_heights, expected[0], atol=5e-5, rtol=0.)
            torch.testing.assert_close(state.foot_terrain_scan_valid, expected[1])
        if args.phase == "geometry":
            record.update(verify_geometry(env, isaac, output, args.play_scan))
        action = obs.new_zeros((4, env.action_dim))
        for _ in range(4):
            env.step(action)
            if isaac:
                free, total = torch.cuda.mem_get_info(device)
                record["sampled_device_used_bytes"] = max(record.get("sampled_device_used_bytes", 0), total - free)
        if isaac:
            torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(24):
            obs, *_ = env.step(action)
            assert torch.isfinite(obs).all()
        if isaac:
            torch.cuda.synchronize()
            record["peak_torch_cuda_bytes"] = torch.cuda.max_memory_allocated()
        record["environment_steps_per_second"] = 96 / (perf_counter() - start)
        record["obs_dim"] = obs.shape[1]
        if args.phase in ("source", "resume"):
            checker = VerifyFinite(runner)
            runner.callbacks.append(checker)
            before = [p.detach().clone() for p in runner.algorithm.policy.parameters()]
            runner.train()
            assert checker.updates and any(not torch.equal(p, old) for p, old in zip(runner.algorithm.policy.parameters(), before))
            if args.phase == "source":
                runner.save(checkpoint)
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                env.load_checkpoint_state_dict(saved["runner"]["environment"])
                env.reset()
                state = env.simulator.get_state()
                torch.save((state.foot_terrain_scan_heights, state.foot_terrain_scan_valid),
                           output / "reset_scan.pt")
            runner.test(num_episodes=4)
            if args.phase == "source":
                runner.play(num_steps=4, formats="gif", num_plays=1, mode="configured")
                assert list(output.rglob("*.gif"))
        print(record, flush=True)
    finally:
        runner.close()
    (output / (args.phase + ".json")).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
