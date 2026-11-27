"""Verify the flat-ground state and scan-off baseline with a real simulator via its normal YAML entry point.

Run from the repository root with the corresponding interpreter:
python -m tests.manual.verify_foot_terrain_scan_plane mujoco
python -m tests.manual.verify_foot_terrain_scan_plane isaac_sim
Add --disabled to verify the scan-off baseline.
On Windows, use the activated Isaac Conda environment. If the host has
conflicting Intel and LLVM OpenMP runtimes, run with MKL_THREADING_LAYER
set to SEQUENTIAL for this verification process.
"""

import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml

from envs.vector_env import VectorEnv
from utils import Component, ComponentInfo, RuntimeContext, load_yaml


ROOT = Path(__file__).resolve().parents[2]
SIMULATORS = {
    "mujoco": "mujoco_unitree_go1_rgb_array.yaml",
    "isaac_sim": "isaac_sim_unitree_go1_headless.yaml",
}


def verify(backend: str, enabled: bool) -> None:
    context = RuntimeContext(torch.device("cpu"), torch.float32, 1, 0, True, ROOT, ROOT)
    config = load_yaml(ROOT / "configs/simulators" / SIMULATORS[backend])
    config.pop("render_mode", None)
    if enabled:
        config["simulator_state_manager_config"] = {"terms": {"foot_terrain_scan": {
            "mount_names": ["scan_rf", "scan_lf", "scan_rh", "scan_lh"],
            "reference_height": .27, "hit_mode": "ground_only", "max_angle_deg": 180.,
        }}}

    with TemporaryDirectory() as directory:
        simulator_path = Path(directory) / "simulator.yaml"
        simulator_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        component = Component(
            None, None, None, None,
            ComponentInfo(backend, simulator_path),
            ComponentInfo("locomotion", ROOT / "configs/tasks/locomotion_unitree_go1_standing.yaml"),
        )
        env = VectorEnv(context)
        calls = []
        if backend == "mujoco":
            from envs.simulators.mujoco import MujocoSimulator as Simulator
        else:
            from envs.simulators.isaac_sim import IsaacSimSimulator as Simulator
        original_query = Simulator.query_surface_rays

        def track_query(simulator, origins, directions, max_distance, env_ids, *, include_robot):
            calls.append(origins.shape[0])
            return original_query(simulator, origins, directions, max_distance, env_ids,
                                  include_robot=include_robot)

        try:
            with patch.object(Simulator, "query_surface_rays", track_query):
                env.config_update(component, num_envs=3, max_episode_steps=32)
                obs = env.reset()
                obs, *_ = env.step(obs.new_zeros((env.num_envs, env.action_dim)))
                sim = env.simulator
                state = sim.get_state().clone()
                env_ids = torch.tensor([2, 0], device=state.qpos.device)
                subset = sim.get_state(env_ids)
                empty = sim.get_state(env_ids[:0])
            assert obs.shape[1] == env.observation_dim
            for name, value in subset.as_dict().items():
                torch.testing.assert_close(value, state.as_dict()[name][env_ids])
            if enabled:
                assert calls and set(calls) == {2, 3}
                assert sim.model_context.state_term_output_shapes["foot_terrain_scan"]["foot_terrain_scan_heights"] == (4, 61)
                valid = state.foot_terrain_scan_valid
                assert valid is not None and valid.all()

                feet_z = state.geom_xpos[:, sim.model_context.foot_geom_ids, 2]
                torch.testing.assert_close(
                    state.foot_terrain_scan_heights, feet_z[..., None].expand(-1, -1, 61),
                    atol=2e-6, rtol=0.,
                )

                empty_heights = empty.foot_terrain_scan_heights
                assert empty_heights is not None and empty_heights.shape == (0, 4, 61)
            else:
                assert env.observation_dim == 112
                assert not calls
                assert not sim.model_context.state_term_output_shapes
                assert not state.term_values
                assert not hasattr(state, "foot_terrain_scan_heights")
            print(f"{backend}: enabled={enabled}, observation_dim={env.observation_dim}, "
                  f"ray_batches={calls}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=SIMULATORS)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()
    verify(args.backend, not args.disabled)
