"""Validate PhysX terrain, PPO resume and window shutdown in separate Isaac processes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
from traceback import print_exc
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from envs.vector_env import VectorEnv
from utils import Component, ComponentInfo, RuntimeContext, load_yaml
from verify_command_curriculum import VerifyFinite, configure
from verify_terrain_curriculum import assert_equal, get_terrain


def build_component(output, num_envs, flat, friction=False):
    environment = output / "environment.yaml"
    environment.write_text(yaml.safe_dump({"num_envs": num_envs, "max_episode_steps": 32}), encoding="utf-8")
    task = output / "task.yaml"
    config = load_yaml(ROOT / "configs/tasks/locomotion_unitree_go1_velocity.yaml")
    if not flat:
        config["randomization_manager_config"]["terms"] = {
            "terrain": {
                "category": {"type": "Curriculum", "params": {
                    "curriculum": "locomotion",
                    "values": {"plane": {}, "slope": {"max_slope": .2}, "rough": {"max_height": .08}},
                }},
                "difficulty": {"type": "Curriculum", "params": {
                    "curriculum": "locomotion", "values": [.5, 1.],
                }},
                "patch_size": [4., 4.], "resolution": .2, "spawn_size": [1.4, 1.2],
            },
        }
        config["termination_manager_config"]["terms"]["terrain_map_bounds"] = {}
        for params in config["command_manager_config"]["terms"].values():
            if "num_cell" in params["params"]:
                params["params"]["num_cell"] = 2
    if friction:
        config.setdefault("randomization_manager_config", {}).setdefault("terms", {})["friction"] = (
            {"type": "Curriculum", "params": {"curriculum": "locomotion", "factors": [.4, 1.2]}}
            if not flat else {"type": "Uniform", "params": {"factors": [.4, 1.2]}}
        )
    task.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return Component(
        runner=None,
        algorithm=ComponentInfo("ppo", ROOT / "configs/algorithms/ppo_num_batch_4.yaml"),
        policy=ComponentInfo("actor_critic", ROOT / "configs/policies/actor_critic_direct_actuation.yaml"),
        environment=ComponentInfo("vector_env", environment),
        simulator=ComponentInfo("isaac_sim", ROOT / "configs/simulators/isaac_sim_unitree_go1_rgb_array.yaml"),
        task=ComponentInfo("locomotion", task),
    )


def verify_physics(environment, flat):
    sim = environment.simulator
    obs = environment.reset()
    assert torch.isfinite(obs).all()
    if not flat:
        from omni.physx import get_physx_scene_query_interface

        lower, upper = sim.get_ground_bounds()
        centres = (lower + upper) / 2
        state = sim.get_state()
        assert state.ground_valid.all() and state.terrain_in_bounds.all()
        assert ((state.qpos[:, :2] - centres).abs() < .7).all()
        environment.set_training_mode(False)
        terrain = sim._require_backend()._terrain
        with patch.object(terrain, "build", side_effect=AssertionError("Reset rebuilt the shared mesh.")):
            environment.reset()
        environment.set_training_mode(True)
        selected = torch.tensor([0], device=centres.device)
        other = torch.tensor([1], device=centres.device)
        snapshot = sim.capture_reset_state(selected)
        untouched = sim.get_state(other).qpos
        query = get_physx_scene_query_interface()

        source = ("plane", 1., 0)
        neighbour = ("slope", .5, 0)
        sim.select_terrain(selected, [source])
        source_bounds = sim.get_ground_bounds(selected)
        sim.select_terrain(selected, [neighbour])
        neighbour_bounds = sim.get_ground_bounds(selected)
        neighbour_centre = (neighbour_bounds[0] + neighbour_bounds[1]) / 2
        sim.select_terrain(selected, [source])
        rough = terrain.origins[terrain._region_ids[("rough", 1., 0)], :2]
        seam = (source_bounds[1] + source_bounds[0] + neighbour_bounds[1] + neighbour_bounds[0]) / 4
        offsets = centres.new_tensor([[1.27, .36], [1.34, .47]])
        border = terrain._map_upper + .5
        points = torch.cat((neighbour_centre + offsets, rough + offsets, seam, border[None]))
        heights = sim.get_ground_heights(points.unsqueeze(0), selected)
        torch.testing.assert_close(heights, sim.get_ground_heights(points.unsqueeze(0), other))
        assert torch.any(heights[0, :4] != 0) and heights[0, -1] == 0.
        for (x, y), height in zip(points.tolist(), heights[0].tolist(), strict=True):
            hit = query.raycast_closest((x, y, 1.), (0., 0., -1.), 2.)
            assert hit["hit"] and hit["collision"] == terrain.prim_path, hit
            assert abs(hit["position"][2] - height) < 2e-5, (hit, height)

        landing = sim.get_state(selected).qpos[:, :3].clone()
        landing[:, :2] = neighbour_centre
        sim.set_root_positions(selected, landing)
        state = sim.get_state(selected)
        assert state.ground_valid.all() and not state.terrain_in_bounds.any()
        torch.testing.assert_close(sim.get_state(other).qpos, untouched)

        runtime = sim._require_backend()
        for _ in range(20):
            sim.step(runtime._ctrl)
            if sim.get_state(selected).foot_ground_contact.any():
                break
        assert sim.get_state(selected).foot_ground_contact.any()
        sim.restore_reset_state(snapshot)
        torch.testing.assert_close(sim.get_ground_bounds(selected)[0], lower[:1])

        pair = torch.cat((selected, other))
        snapshot = sim.capture_reset_state(pair)
        airborne = landing.repeat(2, 1)
        airborne[:, 2] = 2.
        sim.set_root_positions(pair, airborne)
        sim.step(runtime._ctrl)
        overlapping = sim.get_state(selected)
        sim.restore_reset_state(snapshot)
        airborne[1, 0] += 2.
        sim.set_root_positions(pair, airborne)
        sim.step(runtime._ctrl)
        separated = sim.get_state(selected)
        torch.testing.assert_close(overlapping.qpos, separated.qpos, atol=2e-4, rtol=0)
        torch.testing.assert_close(overlapping.qvel, separated.qvel, atol=2e-4, rtol=0)
        sim.restore_reset_state(snapshot)

        outside = terrain._map_upper + terrain.border_width + 1.
        assert torch.isnan(sim.get_ground_heights(outside[None, None], selected)).all()
        environment.reset()
    else:
        assert sim.get_state().ground_valid is None

    actions = obs.new_zeros((environment.num_envs, environment.action_dim))
    torch.cuda.synchronize()
    start = perf_counter()
    foot_contact = False
    for _ in range(64):
        obs, *_ = environment.step(actions)
        assert torch.isfinite(obs).all()
        foot_contact |= bool(sim.get_state().foot_ground_contact.any())
    torch.cuda.synchronize()
    assert foot_contact
    return {
        "num_envs": environment.num_envs,
        "env_steps_per_second": environment.num_envs * 64 / (perf_counter() - start),
        "pytorch_peak_bytes": torch.cuda.max_memory_allocated(),
    }


def verify_human_window(environment):
    from omni.appwindow import get_default_app_window
    from omni.kit.app import get_app
    from omni.kit.viewport.utility import get_active_viewport

    observations = environment.reset()
    actions = observations.new_zeros((environment.num_envs, environment.action_dim))
    for _ in range(5):
        observations, *_ = environment.step(actions)
        environment.simulator.render()
    assert torch.isfinite(observations).all()

    window = get_default_app_window()
    viewport = get_active_viewport()
    assert window is not None and viewport is not None
    manager = get_app().get_extension_manager()
    if environment.simulator._require_backend()._windows_shutdown_workaround:
        for extension in (
            "omni.kit.window.stage",
            "omni.kit.window.property",
            "omni.kit.viewport.menubar.camera",
            "omni.kit.viewport.menubar.display",
            "omni.kit.viewport.menubar.lighting",
            "omni.kit.viewport.menubar.render",
            "omni.kit.viewport.menubar.settings",
        ):
            assert manager.is_extension_enabled(extension), extension
    return window, viewport


def read_foot_friction(environment, env_ids=None):
    runtime = environment.simulator._require_backend()
    view = runtime._simulation_manager.get_physics_simulation_view()
    robot_path = runtime._normalized_robot_prim_path()
    return tuple(
        torch.from_numpy(
            view.create_rigid_body_view(
                f"/World/envs/env_{env_id}{robot_path}{path}"
            ).get_material_properties().numpy().copy()
        )
        for env_id in (range(environment.num_envs) if env_ids is None else env_ids)
        for path in runtime.foot_body_prim_paths
    )


def verify_friction_rollback(environment):
    observations = environment.reset()
    environment.step(observations.new_ones((environment.num_envs, environment.action_dim)))
    term = environment.task.randomization_manager.terms["friction"]
    before = deepcopy(environment.checkpoint_state_dict())
    checked = (0, environment.num_envs - 1)
    material = read_foot_friction(environment, checked)
    state = environment.simulator.get_state()
    selected = torch.zeros(environment.num_envs, device=observations.device, dtype=torch.bool)
    selected[0] = True

    def fail_observation(*args, **kwargs):
        raise RuntimeError("injected observation failure")

    with (
        patch.object(term, "_sample", return_value=[.1]),
        patch.object(environment.task, "compute_observation", side_effect=fail_observation),
    ):
        try:
            environment._reset_done_envs(observations, selected, torch.zeros_like(selected))
        except RuntimeError as error:
            assert str(error) == "injected observation failure"
        else:
            raise AssertionError("The injected reset failure was not propagated.")
    assert_equal(before, environment.checkpoint_state_dict())
    assert_equal(material, read_foot_friction(environment, checked))
    restored = environment.simulator.get_state()
    torch.testing.assert_close(state.qpos, restored.qpos)
    torch.testing.assert_close(state.qvel, restored.qvel)
    print("ISAAC FRICTION ROLLBACK PASSED", flush=True)


def verify_friction_stage_change(environment, component, output):
    config = load_yaml(component.task.config)
    del config["randomization_manager_config"]["terms"]["friction"]
    task_path = output / "task_without_friction.yaml"
    task_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    stage_component = replace(component, simulator=None)
    environment.config_update(replace(
        stage_component, task=replace(component.task, config=task_path),
    ))
    environment.reset()
    for coefficients in read_foot_friction(environment, [0]):
        torch.testing.assert_close(coefficients[..., :2], torch.full_like(coefficients[..., :2], .5))
    environment.config_update(stage_component)
    observations = environment.reset()
    assert torch.isfinite(observations).all()
    assert any((coefficients[..., :2] != .5).any() for coefficients in read_foot_friction(environment, [0]))
    print("ISAAC FRICTION STAGE CHANGE PASSED", flush=True)


def reset_record(environment):
    environment.reset()
    sim = environment.simulator
    root = sim.get_state().qpos[:, :3]
    low, high = sim.get_ground_bounds()
    centres = (low + high) / 2
    offsets = centres.new_tensor([[0., 0.], [1.25, .25], [-1.36, .47]])
    ground = sim.get_ground_heights(centres[:, None, :] + offsets)
    record = {
        "state": deepcopy(environment.checkpoint_state_dict()),
        "patches": get_terrain(environment).patches,
        "root": root.clone(),
        "ground": ground.clone(),
    }
    if "friction" in environment.task.randomization_manager.terms:
        record["friction"] = read_foot_friction(environment)
    return record


def verify_training(output, context, component, phase, render):
    checkpoint = output / "terrain.pt"
    runner = configure(context, component, 2 if phase == "source" else 4,
                       None if phase == "source" else checkpoint)
    try:
        environment = runner.environment
        checker = VerifyFinite(runner)
        runner.callbacks.append(checker)
        if phase == "source":
            runner.train()
            print("ISAAC INITIAL PPO UPDATES PASSED", flush=True)
            terrain = get_terrain(environment)
            terrain.request_reallocation(torch.tensor([0], device=context.device))
            terrain.request_spawn(torch.tensor([1], device=context.device))
            before = deepcopy(environment.checkpoint_state_dict())
            runner.save(checkpoint)
            runner.test(num_episodes=environment.num_envs)
            assert_equal(before, environment.checkpoint_state_dict())
            print("ISAAC FIXED EVALUATION PASSED", flush=True)
            if render:
                from PIL import Image, ImageSequence
                import numpy as np

                videos_before = set(output.rglob("*.gif"))
                runner.play(num_steps=5, formats="gif", num_plays=1, mode="configured")
                assert_equal(before, environment.checkpoint_state_dict())
                videos = set(output.rglob("*.gif")) - videos_before
                assert len(videos) == 1
                with Image.open(videos.pop()) as video:
                    assert any(np.asarray(frame.convert("RGB")).std() > 1 for frame in ImageSequence.Iterator(video))
                print("ISAAC RGB PLAYBACK PASSED", flush=True)
            environment.set_training_mode(True)
            environment.load_checkpoint_state_dict(before)
            torch.save([reset_record(environment) for _ in range(5)], output / "future.pt")
        else:
            before = deepcopy(environment.checkpoint_state_dict())
            records = torch.load(output / "future.pt", weights_only=False)
            for expected in records:
                record = reset_record(environment)
                assert_equal(record, expected)
            print("ISAAC CROSS-PROCESS RESET RECORDS MATCH", flush=True)
            environment.load_checkpoint_state_dict(before)
            parameters = [p.detach().clone() for p in runner.algorithm.policy.parameters()]
            runner.train()
            assert any(not torch.equal(p, before) for p, before in zip(runner.algorithm.policy.parameters(), parameters))
            print("ISAAC RESUMED PPO UPDATES PASSED", flush=True)
        assert checker.updates == 2
        return {"updates": checker.updates, "render": render and phase == "source"}
    except Exception:
        print_exc()
        raise
    finally:
        runner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["physics", "flat", "source", "resume", "human"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--friction", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 2:
        parser.error("--num-envs must be at least 2")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    context = RuntimeContext(torch.device("cuda:0"), torch.float32, 4, 0, False, ROOT, output)
    component = build_component(output, args.num_envs, args.phase in ("flat", "human"), args.friction)
    if args.phase == "human":
        simulator = component.simulator
        config = load_yaml(simulator.config)
        config.update(render_mode="human", train_render_mode="human")
        simulator_path = output / "simulator.yaml"
        simulator_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        component = replace(component, simulator=replace(simulator, config=simulator_path))
    if args.phase in ("source", "resume"):
        result = verify_training(output, context, component, args.phase, args.render)
    else:
        environment = VectorEnv(context)
        try:
            environment.config_update(component, num_envs=args.num_envs, max_episode_steps=32)
            if args.phase == "human":
                # Keep both native wrappers referenced until after Kit shutdown.
                window_handles = verify_human_window(environment)
                result = {"window": True, "viewport": True}
            else:
                result = verify_physics(environment, args.phase == "flat")
                if args.friction:
                    verify_friction_rollback(environment)
                    if args.phase == "flat":
                        verify_friction_stage_change(environment, component, output)
                print("ISAAC PHYSICS CHECKS PASSED", result, flush=True)
        except Exception:
            print_exc()
            raise
        finally:
            environment.close()
        if args.phase == "human":
            del window_handles
    result["closed"] = True
    (output / f"{args.phase}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ISAAC TERRAIN {args.phase.upper()} PASSED", result, flush=True)


if __name__ == "__main__":
    main()
