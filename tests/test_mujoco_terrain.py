# pyright: reportAttributeAccessIssue=false
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import mujoco
import numpy as np
import pytest
import torch

from envs.simulators.mujoco import MujocoSimulator
from envs.tasks.managers.randomization.terms.terrain import TerrainConfig, TerrainRandomizationTerm, TerrainAssignment
from envs.vector_env import VectorEnv
from utils import Component, ComponentInfo


pytestmark = pytest.mark.mujoco
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TerrainScenario:
    """Test environment: it owns physics reset, not the task randomization term."""
    simulator: MujocoSimulator
    term: TerrainRandomizationTerm

    @property
    def patches(self):
        patches = self.term.patches
        assert all(p is not None for p in patches)
        return cast(tuple[TerrainAssignment, ...], patches)

    @property
    def assignments(self):
        return self.term.assignments

    def reset(self, env_ids=None, difficulty_indices=None):
        try:
            self.term.before_simulator_reset(env_ids, difficulty_indices)
            self.simulator.reset(env_ids)
            self.term.after_simulator_reset(env_ids)
            self.term.commit_simulator_reset(env_ids)
        except Exception:
            self.term.abort_simulator_reset(env_ids)
            raise


def configure(context, *, num_envs=3, terrain=None, workers=1, model_path=None,
              feet=("FR", "FL", "RR", "RL"), keyframe: str | None = "home"):
    config = TerrainConfig.from_mapping(terrain if terrain is not None else dict(
        types=["plane", "slope", "rough"], max_height=0.04, max_slope=0.15,
        spawn_size=[1, 1], seed=0))
    manager = TerrainRandomizationTerm(num_envs, context, **asdict(config))
    simulator = MujocoSimulator(context)
    simulator.config_update(
        Component(None, None, None, None, None, None), num_envs=num_envs,
        model_path=model_path or ROOT / "assets/unitree_go1/MJCF/scene.xml",
        sim_dt=0.002, frame_skip=10, step_workers=workers,
        foot_geom_names=feet, floor_geom_names=("floor",), reset_keyframe=keyframe,
        terrain_geometry=manager.geometry,
    )
    manager.bind(simulator)
    return TerrainScenario(simulator, manager)



@pytest.fixture
def scenario(runtime_context):
    sim = configure(runtime_context)
    sim.reset()
    yield sim
    sim.simulator.close()


def integration_state(model, data):
    mask = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, mask))
    mujoco.mj_getState(model, data, state, mask)
    return state


def test_mixed_geometry_keeps_floor_id_and_independent_buffers(scenario):
    assert scenario.simulator._terrain is not None
    assert [p.terrain_type for p in scenario.patches] == ["plane", "slope", "rough"]
    floor_id = scenario.simulator.model_context.floor_geom_ids.item()
    assert floor_id == mujoco.mj_name2id(mujoco.MjModel.from_xml_path(str(scenario.simulator.model_path)),
                                        mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for model in scenario.simulator.models:
        assert model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_HFIELD
    for a, b in zip(scenario.simulator.models, scenario.simulator.models[1:]):
        assert not np.shares_memory(a.hfield_data, b.hfield_data)
    assert np.ptp(scenario.simulator._terrain.grids[0]) == 0
    assert np.ptp(scenario.simulator._terrain.grids[1]) > 0
    assert np.ptp(scenario.simulator._terrain.grids[2]) > 0


def test_height_queries_match_native_collision_rays(scenario):
    assert scenario.simulator._terrain is not None
    points = np.random.default_rng(42).uniform(-1.99, 1.99, (3, 200, 2))
    # Include both diagonal halves, grid edges, vertices and patch corners.
    points[:, :5] = [[-2, -2], [2, 2], [-2, 2], [2, -2], [0, 0]]
    query = torch.tensor(points, dtype=torch.float64)
    actual = scenario.simulator.get_ground_heights(query).numpy()
    for env_id, (model, data) in enumerate(zip(scenario.simulator.models, scenario.simulator.datas)):
        for index, xy in enumerate(points[env_id]):
            height = 2.0 - mujoco.mj_rayHfield(model, data, scenario.simulator._terrain.geom_id,
                np.r_[xy, 2.0], np.array([0., 0., -1.]))
            assert actual[env_id, index] == pytest.approx(height, abs=1e-8)
    subset = scenario.simulator.get_ground_heights(query[[2, 0]], torch.tensor([2, 0]))
    torch.testing.assert_close(subset, torch.tensor(actual[[2, 0]]))
    empty = scenario.simulator.get_ground_heights(torch.empty(0, 4, 2), torch.empty(0, dtype=torch.long))
    assert empty.shape == (0, 4)


@pytest.mark.parametrize("xy, ids", [
    (torch.zeros(3, 2), None), (torch.zeros(1, 2, 2), None),
    (torch.zeros(1, 2, 2), torch.tensor([-1])),
    (torch.zeros(1, 2, 2), torch.tensor([3])),
    (torch.zeros(1, 2, 2), torch.tensor([0.])),
    (torch.full((3, 2, 2), float("nan")), None),
    (torch.full((3, 2, 2), 2.01), None),
])
def test_invalid_ground_queries(scenario, xy, ids):
    with pytest.raises(ValueError):
        scenario.simulator.get_ground_heights(xy, ids)


def test_local_reset_changes_only_selected_environment(scenario):
    action = scenario.simulator.model_context.actuator_default_ctrl.repeat(3, 1)
    for _ in range(10):
        scenario.simulator.step(action)
    states = [integration_state(m, d) for m, d in zip(scenario.simulator.models, scenario.simulator.datas)]
    patches = scenario.patches
    hfields = [m.hfield_data.copy() for m in scenario.simulator.models]
    models, datas = list(scenario.simulator.models), list(scenario.simulator.datas)
    scenario.reset(torch.tensor([0]))
    assert scenario.patches[0] is patches[0]
    np.testing.assert_array_equal(scenario.simulator.models[0].hfield_data, hfields[0])
    for i in (1, 2):
        np.testing.assert_array_equal(integration_state(scenario.simulator.models[i], scenario.simulator.datas[i]), states[i])
        np.testing.assert_array_equal(scenario.simulator.models[i].hfield_data, hfields[i])
        assert scenario.patches[i] == patches[i]
    assert scenario.simulator.models == models and scenario.simulator.datas == datas
    scenario.reset(torch.empty(0, dtype=torch.long))
    assert scenario.patches[1:] == patches[1:]


def test_reset_difficulty_and_invalid_updates_are_atomic(runtime_context):
    sim = configure(runtime_context, num_envs=2, terrain=dict(types=["rough"],
        max_height=.05, spawn_size=[1, 1], difficulty_levels=[0, 1]))
    try:
        assert sim.simulator._terrain is not None
        assert sim.assignments is not None
        sim.reset()
        initial_other = sim.patches[1]
        sim.reset(torch.tensor([0]), difficulty_indices=torch.tensor([1]))
        assert sim.patches[0].difficulty == 1
        assert np.ptp(sim.simulator._terrain.grids[0]) > .01
        assert sim.patches[1] == initial_other
        state = sim.assignments.state_dict()
        arrays = [m.hfield_data.copy() for m in sim.simulator.models]
        for ids, levels in [(torch.tensor([0, 0]), None), (torch.tensor([-1]), None),
                            (torch.tensor([0]), torch.tensor([2])),
                            (torch.tensor([0]), torch.tensor([1, 0]))]:
            with pytest.raises(ValueError):
                sim.reset(ids, difficulty_indices=levels)
            assert sim.assignments.state_dict() == state
            for model, array in zip(sim.simulator.models, arrays):
                np.testing.assert_array_equal(model.hfield_data, array)
    finally:
        sim.simulator.close()


def test_seed_override_and_reset_order_reproduce_geometry_and_spawn(runtime_context):
    first = configure(runtime_context)
    second = configure(replace(runtime_context, seed=999))
    try:
        first.reset(torch.tensor([2, 0, 1]))
        second.reset(torch.tensor([1, 2, 0]))
        for _ in range(3):
            assert first.patches == second.patches
            for m1, d1, m2, d2 in zip(first.simulator.models, first.simulator.datas, second.simulator.models, second.simulator.datas):
                np.testing.assert_array_equal(m1.hfield_data, m2.hfield_data)
                np.testing.assert_array_equal(d1.qpos, d2.qpos)
            first.reset(torch.tensor([2, 0]))
            second.reset(torch.tensor([0, 2]))
    finally:
        first.simulator.close()
        second.simulator.close()


def test_spawn_collision_footprint_stays_in_flat_safe_area(scenario):
    for _ in range(15):
        scenario.reset()
        for model, data in zip(scenario.simulator.models, scenario.simulator.datas):
            robot_ids = np.flatnonzero((model.geom_bodyid != 0) &
                ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
            for i in robot_ids:
                rotation = data.geom_xmat[i].reshape(3, 3)
                centre = data.geom_xpos[i] + rotation @ model.geom_aabb[i, :3]
                radius = np.abs(rotation) @ model.geom_aabb[i, 3:]
                assert np.all(np.abs(centre[:2]) + radius[:2] <= .5 + 1e-7)
                assert centre[2] - radius[2] >= .002 - 1e-7
            for contact in data.contact:
                assert contact.dist >= -1e-7


@pytest.mark.parametrize("kind", ["plane", "slope", "rough"])
def test_actual_foot_contacts_on_nonflat_surface(runtime_context, tmp_path, kind):
    path = tmp_path / "probe.xml"
    path.write_text('''<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1"/>
        <body pos="0 0 .2"><freejoint/><geom name="foot" type="sphere" size=".025" mass="1"/>
        </body></worldbody></mujoco>''', encoding="utf-8")
    sim = configure(runtime_context, num_envs=1, model_path=path, feet=("foot",),
        keyframe=None, terrain=dict(types=[kind, "plane"] if kind != "plane" else ["plane", "rough"],
        weights=[1e100, 1], max_height=.04, max_slope=.1, spawn_size=[.5, .5], seed=42))
    try:
        sim.reset()
        assert sim.patches[0].terrain_type == kind
        xy = torch.tensor([[[1.23, .74]]], dtype=torch.float64)
        height = sim.simulator.get_ground_heights(xy).item()
        if kind != "plane":
            assert abs(height) > 1e-5
        data = sim.simulator.datas[0]
        data.qpos[:3] = [1.23, .74, height + .019]
        mujoco.mj_forward(sim.simulator.models[0], data)
        state = sim.simulator.get_state()
        assert state.foot_ground_contact[0, 0], state.foot_contact_normal_force
        assert state.foot_contact_normal_force[0, 0] > 15
        pairs = state.contact_geom_ids[0].tolist()
        assert any(sim.simulator.model_context.floor_geom_ids.item() in pair for pair in pairs)
        data.qpos[2] = height + .2
        mujoco.mj_forward(sim.simulator.models[0], data)
        assert not sim.simulator.get_state().foot_ground_contact[0, 0]
    finally:
        sim.simulator.close()


def test_parallel_terrain_stepping_matches_serial(runtime_context):
    serial, parallel = configure(runtime_context), configure(runtime_context, workers=3)
    try:
        serial.reset()
        parallel.reset()
        action = serial.simulator.model_context.actuator_default_ctrl.repeat(3, 1)
        for _ in range(30):
            serial.simulator.step(action)
            parallel.simulator.step(action)
        for m, a, b in zip(serial.simulator.models, serial.simulator.datas, parallel.simulator.datas):
            np.testing.assert_array_equal(integration_state(m, a), integration_state(m, b))
        assert torch.all(serial.simulator.get_state().foot_ground_contact)
    finally:
        serial.simulator.close()
        parallel.simulator.close()


def test_legacy_plane_is_unchanged(runtime_context):
    implicit = configure(runtime_context, terrain={})
    explicit = configure(runtime_context, terrain={"types": ["plane"]})
    try:
        implicit.reset()
        explicit.reset()
        action = implicit.simulator.model_context.actuator_default_ctrl.repeat(3, 1)
        for _ in range(30):
            implicit.simulator.step(action)
            explicit.simulator.step(action)
        assert not implicit.patches and not explicit.patches
        for m, a, b in zip(implicit.simulator.models, implicit.simulator.datas, explicit.simulator.datas):
            np.testing.assert_array_equal(integration_state(m, a), integration_state(m, b))
    finally:
        implicit.simulator.close()
        explicit.simulator.close()


def test_too_small_safe_spawn_area_fails_before_first_reset(runtime_context):
    with pytest.raises(ValueError, match="collision footprint"):
        configure(runtime_context, terrain=dict(types=["rough"], max_height=.1))


@pytest.mark.parametrize("env_index", [0, 2])
def test_reset_invalidates_only_changed_rendered_terrain(rollback_scenario, monkeypatch, env_index):
    scenario = rollback_scenario
    monkeypatch.setattr(MujocoSimulator, "playback_env_index", property(lambda self: env_index))
    scenario.simulator._terrain_render_dirty = False
    events = []
    scenario.simulator.renderer = SimpleNamespace(close=lambda: events.append("renderer.close"))
    scenario.simulator.viewer = SimpleNamespace(lock=nullcontext,
        update_hfield=lambda index: events.append(("viewer.upload", index)),
        close=lambda: events.append("viewer.close"))
    scenario.reset(torch.tensor([1]))
    assert events == []
    assert not scenario.simulator._terrain_render_dirty
    scenario.reset(torch.tensor([env_index]))
    assert events == []
    scenario.reset(torch.tensor([env_index]), torch.tensor([1]))
    assert events == [("viewer.upload", scenario.simulator._terrain.hfield_id)]
    assert scenario.simulator.renderer is not None
    assert scenario.simulator._terrain_render_dirty


def test_triangle_support_is_aggregated_before_threshold(scenario):
    foot = scenario.simulator.model_context.foot_geom_ids[0].item()
    floor = scenario.simulator.model_context.floor_geom_ids[0].item()
    ids = torch.tensor([[[foot, floor], [floor, foot]]])
    forces = torch.zeros(1, 2, 6)
    forces[0, :, 0] = 10
    state = scenario.simulator._get_foot_contact_state(ids, forces)
    assert state["foot_ground_contact"][0, 0]
    assert state["foot_contact_normal_force"][0, 0] == 20


def test_translated_floor_heights_and_spawn(runtime_context, tmp_path):
    path = tmp_path / "translated.xml"
    path.write_text('''<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1" pos="4 -3 .2"/>
        <body pos="0 0 .3"><freejoint/><geom name="foot" type="sphere" size=".025" mass="1"/>
        </body></worldbody></mujoco>''', encoding="utf-8")
    sim = configure(runtime_context, num_envs=1, model_path=path, feet=("foot",),
        keyframe=None, terrain=dict(types=["rough"], max_height=.04, spawn_size=[.5, .5]))
    try:
        sim.reset()
        position = sim.simulator.datas[0].qpos[:3]
        assert abs(position[0] - 4) <= .225
        assert abs(position[1] + 3) <= .225
        assert position[2] >= .225
        assert sim.simulator.get_ground_heights(torch.tensor([[[4., -3.]]])).item() == pytest.approx(.2)
    finally:
        sim.simulator.close()


@pytest.mark.parametrize("floor, feet", [
    ('<geom name="floor" type="plane" size="3 3 .1" quat=".92388 .38268 0 0"/>', ("foot",)),
    ('<geom name="floor" type="plane" size="3 3 .1"/><geom type="plane" size="3 3 .1"/>', ("foot",)),
    ('<geom name="floor" type="box" size="3 3 .1"/>', ("foot",)),
])
def test_unsupported_ground_fails_early(runtime_context, tmp_path, floor, feet):
    path = tmp_path / "invalid.xml"
    path.write_text('<mujoco><worldbody>' + floor + '''<body pos="0 0 .3"><freejoint/>
        <geom name="foot" type="sphere" size=".025" mass="1"/></body></worldbody></mujoco>''', encoding="utf-8")
    # Empty explicit names allow default '*foot' discovery, so remove the suffix.
    if not feet:
        path.write_text(path.read_text().replace('name="foot"', 'name="ball"'), encoding="utf-8")
    with pytest.raises(ValueError):
        configure(runtime_context, model_path=path, keyframe=None, feet=feet,
            terrain=dict(types=["rough"], max_height=.04, spawn_size=[.5, .5]))


def test_backend_reset_executes_without_sampling_terrain(scenario):
    assert scenario.assignments is not None
    state = scenario.assignments.state_dict()
    patches = scenario.patches
    heights = [model.hfield_data.copy() for model in scenario.simulator.models]
    scenario.simulator.reset(torch.tensor([0]))
    assert scenario.assignments.state_dict() == state
    assert scenario.patches == patches
    for model, original in zip(scenario.simulator.models, heights):
        np.testing.assert_array_equal(model.hfield_data, original)
    np.testing.assert_array_equal(scenario.simulator.datas[0].qpos,
        scenario.simulator.models[0].key_qpos[scenario.simulator.reset_keyframe_id])


def test_geometry_application_does_not_reset_robot_or_allocate(scenario):
    assert scenario.assignments is not None
    backend = scenario.simulator
    states = [integration_state(m, d) for m, d in zip(backend.models, backend.datas)]
    allocation = scenario.assignments.state_dict()
    patch = replace(scenario.patches[0].geometry, heights=tuple(tuple(.01 for _ in row)
                                                     for row in scenario.patches[0].geometry.heights))
    backend.apply_terrain(env_ids_tensor=torch.tensor([0]), patches=[patch])
    for model, data, original in zip(backend.models, backend.datas, states):
        np.testing.assert_array_equal(integration_state(model, data), original)
    assert scenario.assignments.state_dict() == allocation
    assert backend.get_ground_heights(torch.zeros(1, 1, 2), torch.tensor([0])).item() == pytest.approx(.01, abs=1e-7)


def test_invalid_patch_batch_does_not_partially_update_geometry(scenario):
    old = [model.hfield_data.copy() for model in scenario.simulator.models]
    invalid = replace(scenario.patches[1].geometry, heights=((999.,),))
    with pytest.raises(ValueError, match="collision geometry"):
        scenario.simulator.apply_terrain(torch.tensor([0, 1]), [scenario.patches[0].geometry, invalid])
    for model, heights in zip(scenario.simulator.models, old):
        np.testing.assert_array_equal(model.hfield_data, heights)


def test_task_randomization_owns_terrain_and_routes_full_and_done_resets(runtime_context):
    env = VectorEnv(replace(runtime_context, load_dir=ROOT, save_dir=ROOT))
    component = Component(None, None, None, None,
        ComponentInfo("mujoco", ROOT / "configs/simulators/mujoco_unitree_go1_rgb_array.yaml"),
        ComponentInfo("locomotion", ROOT / "configs/tasks/locomotion_unitree_go1_standing_terrain.yaml"))
    try:
        env.config_update(component, num_envs=3, max_episode_steps=100)
        assert env.task.randomization_manager is not None
        manager = cast(TerrainRandomizationTerm, env.task.randomization_manager.terms["terrain"])
        assert manager is not None and manager.assignments is not None
        assert manager.simulator is env.simulator
        observation = env.reset()
        assert observation.shape == (3, 112)
        previous = manager.patches
        generations = manager.assignments.generations.copy()
        env.current_episode_steps[1] = 99
        env.step(torch.zeros(3, 12))
        assert manager.assignments.generations == generations
        assert manager.patches == previous
        assert manager.patches[0] == previous[0] and manager.patches[2] == previous[2]
        assert env.current_episode_steps.tolist() == [1, 0, 1]
        physics = [integration_state(m, d) for m, d in zip(env.simulator.models, env.simulator.datas)]
        allocation = manager.assignments.state_dict()
        env.task.reset(torch.tensor([1]))
        assert manager.assignments.state_dict() == allocation
        for model, data, original in zip(env.simulator.models, env.simulator.datas, physics):
            np.testing.assert_array_equal(integration_state(model, data), original)
        # Switching to a task without terrain removes the collision layout;
        # simulator inheritance must not silently retain the previous heightfield.
        plain_task = Component(None, None, None, None, None,
            ComponentInfo("locomotion", ROOT / "configs/tasks/locomotion_unitree_go1_standing.yaml"))
        env.config_update(plain_task)
        assert env.task.randomization_manager is not None
        assert "terrain" not in env.task.randomization_manager.terms
        assert env.simulator._terrain is None
        env.reset()
        for data in env.simulator.datas:
            np.testing.assert_array_equal(data.qpos[:2], [0, 0])
    finally:
        env.close()


def test_spawn_bounds_do_not_require_named_feet(runtime_context, tmp_path):
    path = tmp_path / "generic_robot.xml"
    path.write_text('''<mujoco><worldbody><geom name="floor" type="plane" size="3 3 .1"/>
        <body pos="0 0 .3"><freejoint/><geom name="ball" type="sphere" size=".025" mass="1"/>
        </body></worldbody></mujoco>''', encoding="utf-8")
    sim = configure(runtime_context, num_envs=1, model_path=path, feet=(), keyframe=None,
        terrain=dict(types=["rough"], max_height=.04, spawn_size=[.5, .5]))
    try:
        assert sim.simulator.model_context.foot_geom_ids.numel() == 0
        sim.reset()
        measured = sim.simulator.get_terrain_spawn_geometry()
        np.testing.assert_allclose(measured.collision_lower, [-.025] * 3)
        np.testing.assert_allclose(measured.collision_upper, [.025] * 3)
        assert np.isfinite(sim.simulator.datas[0].qpos).all()
    finally:
        sim.simulator.close()


@pytest.mark.parametrize("phase, replace_terrain", [
    ("apply", True), ("physics_reset", True), ("spawn", True),
    ("physics_reset", False), ("spawn", False),
])
@pytest.mark.parametrize("partial", [False, True], ids=["full", "partial"])
def test_failed_environment_reset_restores_physics_and_allocation(rollback_scenario, runtime_context, monkeypatch, phase, partial, replace_terrain):
    scenario = rollback_scenario
    from envs.tasks import BaseTaskLogic
    from envs.tasks.managers.randomization import RandomizationManager
    backend, term = scenario.simulator, scenario.term
    manager = RandomizationManager(3, runtime_context)
    manager.terms["terrain"] = term
    manager.bind(backend)
    assert term.assignments is not None and backend._terrain is not None
    action = backend.model_context.actuator_default_ctrl.repeat(3, 1)
    for _ in range(4):
        backend.step(action)
    ids = torch.tensor([1]) if partial else torch.arange(3)
    if replace_terrain:
        term.set_difficulty(env_ids_tensor=ids, difficulty_indices=torch.ones(len(ids), dtype=torch.long))
    assignments, patches = term.assignments.state_dict(), term.patches
    states = [integration_state(m, d) for m, d in zip(backend.models, backend.datas)]
    accelerations = [d.qacc.copy() for d in backend.datas]
    heights = [m.hfield_data.copy() for m in backend.models]
    grids = [g.copy() for g in backend._terrain.grids]
    original_reset, original_spawn, original_install = backend.reset, backend.set_root_positions, backend._terrain.install

    def fail_reset(selected=None):
        original_reset(selected)
        raise RuntimeError("injected physics reset failure")

    def fail_spawn(selected, positions):
        original_spawn(selected, positions)
        raise RuntimeError("injected spawn failure")

    def fail_install(model, patch):
        original_install(model, patch)
        raise RuntimeError("injected apply failure")

    if phase == "apply":
        monkeypatch.setattr(backend._terrain, "install", fail_install)
    elif phase == "physics_reset":
        monkeypatch.setattr(backend, "reset", fail_reset)
    else:
        monkeypatch.setattr(backend, "set_root_positions", fail_spawn)
    env = VectorEnv(runtime_context)
    env.simulator = backend
    env.task = cast(BaseTaskLogic, SimpleNamespace(
        before_simulator_reset=manager.before_simulator_reset,
        after_simulator_reset=manager.after_simulator_reset,
        abort_simulator_reset=manager.abort_simulator_reset))
    env.current_episode_steps = torch.tensor([5, 6, 7])
    with pytest.raises(RuntimeError, match="injected"):
        if partial:
            env._reset_done_envs(torch.zeros(3, 1), torch.tensor([False, True, False]), torch.zeros(3, dtype=torch.bool))
        else:
            env.reset()
    assert term.assignments.state_dict() == assignments and term.patches == patches
    assert env.current_episode_steps.tolist() == [5, 6, 7]
    for i, (model, data) in enumerate(zip(backend.models, backend.datas)):
        np.testing.assert_array_equal(integration_state(model, data), states[i])
        np.testing.assert_array_equal(data.qacc, accelerations[i])
        np.testing.assert_array_equal(model.hfield_data, heights[i])
        np.testing.assert_array_equal(backend._terrain.grids[i], grids[i])
    # Retry advances allocation only when a different difficulty was requested.
    monkeypatch.setattr(backend, "reset", original_reset)
    monkeypatch.setattr(backend, "set_root_positions", original_spawn)
    monkeypatch.setattr(backend._terrain, "install", original_install)
    scenario.reset(ids)
    expected = assignments["generations"].copy()
    for i in ids.tolist():
        expected[i] += int(replace_terrain)
        assert scenario.patches[i].difficulty == (1.0 if replace_terrain else .25)
        if not replace_terrain:
            assert scenario.patches[i] is patches[i]
    assert term.assignments.generations == expected


@pytest.mark.parametrize("env_index", [0, 2])
def test_renderer_upload_preserves_context_and_camera(rollback_scenario, monkeypatch, env_index):
    scenario = rollback_scenario
    monkeypatch.setattr(MujocoSimulator, "playback_env_index", property(lambda self: env_index))
    backend = scenario.simulator
    events = []
    context = SimpleNamespace(make_current=lambda: events.append("current"))
    renderer = SimpleNamespace(_gl_context=context, _mjr_context=object(),
        update_scene=lambda *args, **kwargs: events.append("scene"),
        render=lambda: np.zeros((8, 8, 3), dtype=np.uint8), close=lambda: None)
    camera = object()
    backend.renderer, backend._render_camera = renderer, camera
    def upload(model, *args):
        assert model is backend.models[env_index]
        events.append("upload")

    monkeypatch.setattr(mujoco, "mjr_uploadHField", upload)
    backend._terrain_render_dirty = False
    scenario.reset(torch.tensor([1]))
    backend._rgb_array_render()
    assert events == ["scene"]
    events.clear()
    scenario.reset(torch.tensor([env_index]))
    backend._rgb_array_render()
    assert events == ["scene"]
    events.clear()
    scenario.reset(torch.tensor([env_index]), torch.tensor([1]))
    backend._rgb_array_render()
    assert events == ["current", "upload", "scene"]
    assert backend.renderer is renderer and backend._render_camera is camera
    events.clear()
    backend._rgb_array_render()
    assert events == ["scene"]


@pytest.fixture
def rollback_scenario(runtime_context):
    sim = configure(runtime_context, terrain=dict(types=["rough"], max_height=.05,
        spawn_size=[1, 1], difficulty_levels=[.25, 1]))
    sim.reset()
    yield sim
    sim.simulator.close()


@pytest.mark.parametrize("operation", ["reset", "apply", "positions", "restore"])
def test_viewer_locks_only_playback_environment(scenario, monkeypatch, operation):
    from contextlib import contextmanager
    backend = scenario.simulator
    monkeypatch.setattr(MujocoSimulator, "playback_env_index", property(lambda self: 2))
    locks = []

    @contextmanager
    def lock():
        locks.append("lock")
        yield

    backend.viewer = SimpleNamespace(lock=lock, update_hfield=lambda index: None, close=lambda: None)
    for env_id in [0, 2]:
        locks.clear()
        ids = torch.tensor([env_id])
        if operation == "reset":
            backend.reset(ids)
        elif operation == "apply":
            backend.apply_terrain(ids, [scenario.patches[env_id].geometry])
        elif operation == "positions":
            backend.set_root_positions(ids, torch.tensor(backend.datas[env_id].qpos[:3].copy())[None])
        else:
            snapshot = backend.capture_reset_state(env_ids_tensor=ids)
            backend.restore_reset_state(snapshot)
        assert locks == (["lock"] if env_id == backend.playback_env_index else [])
