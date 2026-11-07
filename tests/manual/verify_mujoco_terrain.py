"""Exercise real Go1 contacts on all stage-2 surfaces, without stage-3 tasks/PPO."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch
from PIL import Image
from envs.simulators.mujoco import MujocoSimulator
from utils import Component, RuntimeContext, load_yaml
from envs.tasks.managers.randomization import RandomizationManager


def save_overview(model, data, path):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [0, 0, 0]
    camera.distance = 5
    camera.azimuth = 130
    camera.elevation = -35
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(path)


def reset_from_environment(simulator, randomization, env_ids=None, difficulty_indices=None):
    if difficulty_indices is not None:
        randomization.terms["terrain"].set_difficulty(env_ids, difficulty_indices)
    try:
        randomization.before_simulator_reset(env_ids)
        simulator.reset(env_ids)
        randomization.after_simulator_reset(env_ids)
    except Exception:
        randomization.abort_simulator_reset(env_ids)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests/manual/results/stage2_mujoco")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.steps < 100:
        parser.error("--steps must be >= 100 to include settling")
    context = RuntimeContext(torch.device("cpu"), torch.float32, 4, 0, True, ROOT, ROOT)
    torch.set_num_threads(4)
    sim = MujocoSimulator(context)
    config = load_yaml(ROOT / "configs/simulators/mujoco_unitree_go1_rgb_array.yaml")
    config["model_path"] = ROOT / config["model_path"]
    task_config = load_yaml(ROOT / "configs/tasks/locomotion_unitree_go1_standing_terrain.yaml")
    randomization = RandomizationManager(3, context, **task_config["randomization_manager_config"])
    manager = randomization.terms["terrain"]
    assert manager is not None
    config["terrain_geometry"] = manager.geometry
    started = time.perf_counter()
    try:
        sim.config_update(Component(None, None, None, None, None, None), num_envs=3, **config)
        randomization.bind(sim)
        initialization = time.perf_counter() - started
        reset_from_environment(sim, randomization)
        assert [p.terrain_type for p in manager.patches] == ["plane", "slope", "rough"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = dict(python=sys.version, mujoco=mujoco.__version__, platform=platform.platform(),
                      measured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      initialization_seconds=initialization, num_envs=3, steps_per_phase=args.steps,
                      config=str(ROOT / "configs/simulators/mujoco_unitree_go1_rgb_array.yaml"),
                      randomization_config=str(ROOT / "configs/tasks/locomotion_unitree_go1_standing_terrain.yaml"),
                      phases=[])
        action = sim.model_context.actuator_default_ctrl.repeat(3, 1)
        for phase in ("safe_spawn", "outside_spawn_on_surface"):
            if phase != "safe_spawn":
                # Keep the same terrain, place each robot on a genuinely nonflat
                # region, then let physics settle with constant home targets.
                for env_id, (model, data) in enumerate(zip(sim.models, sim.datas)):
                    mujoco.mj_resetDataKeyframe(model, data, sim.reset_keyframe_id)
                    data.qpos[:2] = [1.2, 0.]
                    mujoco.mj_forward(model, data)
                    feet_xy = data.geom_xpos[sim.model_context.foot_geom_ids.numpy(), :2]
                    heights = sim.get_ground_heights(torch.tensor(feet_xy[None]), torch.tensor([env_id]))
                    data.qpos[2] = .32 + heights.max().item()
                    mujoco.mj_forward(model, data)
            seen = torch.zeros(3, 4, dtype=torch.bool)
            times = []
            for step in range(args.steps):
                start = time.perf_counter()
                sim.step(action)
                state = sim.get_state()
                times.append((time.perf_counter() - start) * 1000)
                assert torch.isfinite(state.qpos).all()
                if step >= 50:
                    seen |= state.foot_ground_contact
            assert torch.all(seen), (phase, seen)
            assert torch.all(state.foot_ground_contact), (phase, state.foot_contact_normal_force)
            feet_xy = state.geom_xpos[:, sim.model_context.foot_geom_ids, :2]
            ground_heights = sim.get_ground_heights(feet_xy)
            base_ground = sim.get_ground_heights(state.qpos[:, None, :2])[:, 0]
            assert torch.all((state.qpos[:, 2] - base_ground > .15)
                             & (state.qpos[:, 2] - base_ground < .4))
            if phase == "outside_spawn_on_surface":
                for env_id in (1, 2):
                    assert ground_heights[env_id].max() - ground_heights[env_id].min() > 1e-5
            report["phases"].append(dict(name=phase, types=[p.terrain_type for p in manager.patches],
                patch_ids=[p.patch_id for p in manager.patches],
                final_qpos=state.qpos.tolist(), foot_contact_seen=seen.tolist(),
                final_foot_ground_heights=ground_heights.tolist(),
                difficulties=[p.difficulty for p in manager.patches],
                slopes=[p.slope for p in manager.patches],
                final_foot_normal_force=state.foot_contact_normal_force.tolist(),
                mean_step_and_state_ms=float(np.mean(times[50:]))))
        # Exercise a real offscreen context before and after env_0 changes terrain.
        if args.render:
            first = sim.render()
            renderer = sim.renderer
            gl_context = renderer._gl_context
            camera = sim._render_camera
            Image.fromarray(first).save(args.output_dir / "before_reset.png")
            # Show every independent model, including terrain away from the
            # central flat spawn platform, with the same overview camera.
            for env_id, (model, data, patch) in enumerate(zip(sim.models, sim.datas, manager.patches)):
                save_overview(model, data, args.output_dir / f"terrain_{env_id}_{patch.terrain_type}.png")
        # Ordinary episode resets must preserve terrain and avoid GPU uploads.
        patches_before_reuse = manager.patches
        generations_before_reuse = manager.assignments.generations.copy()
        heights_before_reuse = [model.hfield_data.copy() for model in sim.models]
        reset_from_environment(sim, randomization)
        assert all(a is b for a, b in zip(manager.patches, patches_before_reuse))
        assert manager.assignments.generations == generations_before_reuse
        for model, heights in zip(sim.models, heights_before_reuse):
            np.testing.assert_array_equal(model.hfield_data, heights)
        if args.render:
            assert not sim._terrain_render_dirty
        report["episode_reset_reuses_terrain"] = True
        untouched = [d.qpos.copy() for d in sim.datas]
        previous = manager.patches
        old_heights = sim.models[0].hfield_data.copy()
        reset_from_environment(sim, randomization, torch.tensor([0]), difficulty_indices=torch.tensor([2]))
        if args.render:
            # A difficulty change can legitimately select plane again. Require changed
            # geometry here so image verification exercises heightfield upload,
            # rather than only a different robot pose on the same flat surface.
            # Alternate difficulty requests; ordinary resets reuse geometry.
            for attempt in range(16):
                if not np.array_equal(old_heights, sim.models[0].hfield_data):
                    break
                reset_from_environment(sim, randomization, torch.tensor([0]),
                    difficulty_indices=torch.tensor([0 if attempt % 2 == 0 else 2]))
            assert not np.array_equal(old_heights, sim.models[0].hfield_data)
        for i in (1, 2):
            np.testing.assert_array_equal(sim.datas[i].qpos, untouched[i])
            assert manager.patches[i] == previous[i]
        if args.render:
            second = sim.render()
            assert sim.renderer is renderer and sim._render_camera is camera
            assert sim.renderer._gl_context is gl_context
            assert not sim._terrain_render_dirty
            Image.fromarray(second).save(args.output_dir / "after_reset.png")
            save_overview(sim.models[0], sim.datas[0], args.output_dir / "terrain_after_reset.png")
            assert first.shape == second.shape and np.any(first != second)
        report["partial_reset_isolated"] = True
        report["render_before_and_after_reset"] = args.render
        if args.render:
            report["rendered_terrain_types"] = [p.terrain_type for p in previous]
            report["rendered_reset_patch_type"] = manager.patches[0].terrain_type
            report["rendered_reset_geometry_changed"] = True
            report["renderer_and_camera_reused"] = True
            report["opengl_context_reused"] = True
        report["passed"] = True
        (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("MUJOCO TERRAIN VERIFICATION PASSED", flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
