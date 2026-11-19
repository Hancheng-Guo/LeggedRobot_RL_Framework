"""Exercise real ANYmal C contacts on all stage-2 surfaces, without stage-3 tasks/PPO."""
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


def reset_from_environment(simulator, randomization, env_ids=None, difficulties=None):
    term = randomization.terms["terrain"]
    env_ids_tensor = torch.arange(term.num_envs) if env_ids is None else env_ids
    if difficulties is None:
        requests = term.difficulty.checkpoint_state_dict()["requests"]
        selected_values = []
        for env_id in env_ids_tensor.tolist():
            patch = term.patches[env_id]
            current = term.config.difficulty.initial_value if patch is None else patch.difficulty
            selected_values.append(requests.get(env_id, current))
        difficulties = torch.tensor(selected_values)
    term.set_difficulty(env_ids_tensor, difficulties)
    try:
        randomization.before_simulator_reset(env_ids)
        simulator.reset(env_ids)
        randomization.after_simulator_reset(env_ids)
        randomization.commit_simulator_reset(env_ids)
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
    config = load_yaml(ROOT / "configs/simulators/mujoco_anymal_c_rgb_array.yaml")
    config["model_path"] = ROOT / config["model_path"]
    task_config = load_yaml(ROOT / "configs/tasks/locomotion_anymal_c_phase.yaml")
    task_config["randomization_manager_config"]["terms"]["terrain"]["category"]["type"] = "Curriculum"
    task_config["randomization_manager_config"]["terms"]["terrain"]["category"]["params"].pop("curriculum", None)
    task_config["randomization_manager_config"]["terms"]["terrain"]["difficulty"]["type"] = "Uniform"
    task_config["randomization_manager_config"]["terms"]["terrain"]["difficulty"]["params"].pop("curriculum", None)
    task_config["randomization_manager_config"]["terms"]["terrain"]["seed"] = 39
    task_config["randomization_manager_config"]["terms"]["terrain"]["category"]["params"]["curriculum"] = "surfaces"

    class StaticCategories:
        # Force all three surfaces for this physical acceptance, independently of learning.
        def get_term(self, name):
            return self

        def get_bounds(self, dimension, env_ids=None):
            env_ids_tensor = torch.arange(3) if env_ids is None else env_ids
            values = (2 - env_ids_tensor % 3).to(context.dtype).reshape(-1, 1)
            return values, values

    randomization = RandomizationManager(
        3, context, curriculum_manager=StaticCategories(), **task_config["randomization_manager_config"],
    )
    manager = randomization.terms["terrain"]
    assert manager is not None
    config["terrain_geometry"] = manager.geometry
    started = time.perf_counter()
    try:
        sim.config_update(Component(None, None, None, None, None, None), num_envs=3, **config)
        randomization.bind(sim)
        initialization = time.perf_counter() - started
        reset_from_environment(sim, randomization)
        assert [p.terrain_type for p in manager.patches] == ["rough", "slope", "plane"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = dict(python=sys.version, mujoco=mujoco.__version__, platform=platform.platform(),
                      measured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      initialization_seconds=initialization, num_envs=3, steps_per_phase=args.steps,
                      config=str(ROOT / "configs/simulators/mujoco_anymal_c_rgb_array.yaml"),
                      randomization_config=str(ROOT / "configs/tasks/locomotion_anymal_c_phase.yaml"),
                      phases=[])
        reset_height = sim.get_state().qpos[:, 2].clone()
        target_height = task_config["constants"]["base_target_height"]
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
                    data.qpos[2] = reset_height[env_id].item() + heights.max().item()
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
            assert torch.all((state.qpos[:, 2] - base_ground > target_height * .5)
                             & (state.qpos[:, 2] - base_ground < target_height * 1.5))
            if phase == "outside_spawn_on_surface":
                for env_id, patch in enumerate(manager.patches):
                    if patch.terrain_type != "plane":
                        assert ground_heights[env_id].max() - ground_heights[env_id].min() > 1e-5
            report["phases"].append(dict(name=phase, patch_ids=[p.patch_id for p in manager.patches], final_qpos=state.qpos.tolist(), foot_contact_seen=seen.tolist(), final_foot_ground_heights=ground_heights.tolist(), difficulties=[p.difficulty for p in manager.patches], final_foot_normal_force=state.foot_contact_normal_force.tolist(), mean_step_and_state_ms=float(np.mean(times[50:])), types=[p.terrain_type for p in manager.patches]))
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
        reset_from_environment(sim, randomization, torch.tensor([0]), difficulties=torch.tensor([1.]))
        if args.render:
            # Environment 0 is deliberately rough so difficulty changes exercise
            # the visible heightfield upload, rather than only moving the robot.
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
