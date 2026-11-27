from dataclasses import replace
from math import cos, pi, sin
from types import SimpleNamespace

import pytest
import torch

from envs.simulators.state.terms.foot_terrain_scan import FootTerrainScan
from envs.simulators.utils.ray_query import GROUND_RAY_QUERY, HorizontalPlaneRayQuery
from envs.simulators.utils.scan_ray import select_nearest_surface
from envs.simulators.state.terms import BaseSimulatorStateTerm, register_simulator_state_term
from envs.simulators.state.terms import registry
from envs.simulators.state import SimulatorStateManager


class _ScanSimulator(SimpleNamespace):
    def query_surface_rays(self, origins, directions, max_distance, env_ids, *, include_robot):
        ground = self.query_ground_rays(origins, directions, max_distance, env_ids)
        robot = self.query_robot_rays(origins, directions, max_distance, env_ids) if include_robot else None
        return select_nearest_surface(ground, robot)


def _mounts(names):
    return torch.zeros(len(names), 3), torch.tensor([[0., 0., -1.]]).repeat(len(names), 1)


def _manager(context, model_context, query, robot_query=None, **config):
    config.setdefault("mount_names", [str(i) for i in model_context.foot_geom_ids.tolist()])
    config.setdefault("reference_height", .5)
    config.setdefault("hit_mode", "ground_only")
    manager = SimulatorStateManager(
        context=context,
        model_context=model_context,
        simulator=_ScanSimulator(query_ground_rays=query, resolve_scan_mounts=_mounts, query_robot_rays=robot_query),
        terms={"foot_terrain_scan": config},
    )
    return manager, manager.model_context


def _model(model_context, foot_ids=(0, 1)):
    return replace(
        model_context,
        base_pos_qpos_ids=torch.arange(3),
        base_quat_qpos_ids=torch.arange(3, 7),
        foot_geom_ids=torch.tensor(foot_ids),
    )


def _read(manager, position, quaternion, feet, env_ids=None):
    count, foot_count = feet.shape[:2]
    empty = feet.new_empty((count, 0))
    return manager.assemble(
        env_ids,
        qpos=torch.cat((position, quaternion), dim=-1),
        qvel=empty, qacc=empty, ctrl=empty,
        geom_xpos=feet, geom_xvel=empty, actuator_force=empty,
        base_lin_vel_body=empty, base_ang_vel_body=empty,
        contact_geom_ids=empty, contact_forces=empty,
        foot_ground_contact=torch.zeros(count, foot_count, dtype=torch.bool),
        foot_contact_normal_force=feet.new_zeros((count, foot_count)),
    )


def test_honeycomb_layout_round_trip(runtime_context, model_context):
    manager, model = _manager(runtime_context, _model(model_context), HorizontalPlaneRayQuery(0.))
    scan = manager.terms["foot_terrain_scan"]
    assert isinstance(scan, FootTerrainScan)
    coordinates = scan.axial_coordinates
    assert model.state_term_output_shapes["foot_terrain_scan"]["foot_terrain_scan_heights"] == (2, 61)
    assert len(set(coordinates)) == 61 and coordinates[0] == (0, 0)
    assert coordinates == tuple(sorted(coordinates, key=lambda p: (max(abs(p[0]), abs(p[1]), abs(sum(p))), p[1], p[0])))
    distances = scan.offsets.norm(dim=-1)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    neighbor_indices = [scan.point_indices[point] for point in neighbors]
    torch.testing.assert_close(distances[neighbor_indices], torch.full((6,), .12))
    assert torch.isclose(distances, torch.tensor(.48)).sum().item() == 6
    torch.testing.assert_close(distances.max(), torch.tensor(.48))
    q, r = torch.tensor(coordinates).T + 4
    heights = torch.arange(61)
    valid = heights % 3 != 0
    layout = torch.zeros((9, 9), dtype=torch.bool)
    height_grid = torch.zeros((9, 9), dtype=heights.dtype)
    valid_grid = torch.zeros_like(layout)
    layout[r, q] = True
    height_grid[r, q] = heights
    valid_grid[r, q] = valid
    assert layout.sum().item() == 61
    assert torch.equal(height_grid[r, q], heights)
    assert torch.equal(valid_grid[r, q], valid)


@pytest.mark.parametrize("params", [
    {"spacing": 0}, {"spacing": float("nan")}, {"spacing": True},
    {"layers": -1}, {"layers": 1.5}, {"layers": True}, {"layers": 1000},
    {"reference_height": 0}, {"hit_mode": "visible"}, {"max_angle_deg": 190},
    {"mount_names": ["same", "same"]}, {"mount_names": []},
    {"max_distance": float("inf")}, {"max_distance": -1},
])
def test_scan_config_validation(runtime_context, model_context, params):
    with pytest.raises(ValueError, match="foot_terrain_scan"):
        _manager(runtime_context, model_context, HorizontalPlaneRayQuery(0.), **params)


def test_scan_rejects_unknown_config_parameter(runtime_context, model_context):
    with pytest.raises(TypeError, match="origin_height"):
        _manager(runtime_context, model_context, HorizontalPlaneRayQuery(0.), origin_height=.6)


def test_disabled_manager_preserves_existing_fields(runtime_context, model_context):
    model = _model(model_context)
    calls = []

    def query(origins, directions, max_distance, env_ids):
        calls.append(origins.shape)
        return HorizontalPlaneRayQuery(0.)(origins, directions, max_distance, env_ids)

    enabled, scan_model = _manager(runtime_context, model, query)
    def unexpected_query(*args):
        pytest.fail("A disabled scan must not query ground rays.")

    disabled = SimulatorStateManager(
        runtime_context, scan_model, simulator=SimpleNamespace(query_ground_rays=unexpected_query),
    )
    assert not disabled.model_context.state_term_output_shapes
    position = torch.tensor([[0., 0., .5]])
    quaternion = torch.tensor([[1., 0., 0., 0.]])
    feet = torch.tensor([[[0., .2, .1], [0., -.2, .3]]])
    before = _read(disabled, position, quaternion, feet)
    assert not before.term_values
    assert not calls
    after = _read(enabled, position, quaternion, feet)
    assert calls == [torch.Size([1, 2, 3])]
    for name, value in before.as_dict().items():
        assert torch.equal(after.as_dict()[name], value)


def test_registered_term_supplies_its_own_state_fields(runtime_context, model_context, monkeypatch):
    monkeypatch.setattr(
        registry, "SIMULATOR_STATE_TERM_CLASSES", dict(registry.SIMULATOR_STATE_TERM_CLASSES),
    )

    simulator = object()

    @register_simulator_state_term
    class BaseHeightProbe(BaseSimulatorStateTerm):
        output_shapes = {"height": (1,), "unobserved_height": (1,)}

        def __init__(self, name, *, simulator, **kwargs):
            super().__init__(name, **kwargs)
            self.simulator = simulator

        def compute(self, state, env_ids):
            return {"height": state.qpos[:, 2:3], "unobserved_height": None}

    SimulatorStateManager.validate_config(
        unavailable_capabilities={GROUND_RAY_QUERY: "Ground ray queries unavailable."},
        terms={"base_height_probe": {}},
    )
    manager = SimulatorStateManager(
        runtime_context, _model(model_context),
        terms={"base_height_probe": {}}, simulator=simulator,
    )
    term = manager.terms["base_height_probe"]
    assert isinstance(term, BaseHeightProbe) and term.simulator is simulator
    position = torch.tensor([[0., 0., .5], [0., 0., .7]])
    quaternion = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
    state = _read(manager, position, quaternion, position[:, None])
    assert manager.model_context.state_term_output_shapes == {"base_height_probe": term.output_shapes}
    assert state.unobserved_height is None
    snapshot = state.clone().select(torch.tensor([1]))
    height = state.height
    assert height is not None
    height.zero_()
    torch.testing.assert_close(
        snapshot.as_dict()["height"], torch.tensor([[.7]]),
    )

    @register_simulator_state_term
    class OtherHeightProbe(BaseHeightProbe):
        pass

    with pytest.raises(ValueError, match="other_height_probe.*duplicate fields.*height"):
        SimulatorStateManager(
            runtime_context, _model(model_context),
            terms={"base_height_probe": {}, "other_height_probe": {}}, simulator=simulator,
        )


def test_plane_ray_query_includes_endpoints_and_rejects_parallel_rays():
    origins = torch.tensor([[[0., 0., .5], [0., 0., 2.5], [0., 0., 2.51]]])
    directions = torch.tensor([[0., 0., -1.], [1., 0., 0.]]).expand(1, 3, 2, 3)
    distance = HorizontalPlaneRayQuery(.5)(origins, directions, 2., None)
    torch.testing.assert_close(distance[0, :2, 0], torch.tensor([0., 2.]))
    assert distance[0, 2, 0].isnan() and distance[..., 1].isnan().all()


@pytest.mark.parametrize("mode", ["first_surface", "visible_ground", "ground_only"])
def test_moving_occluder_keeps_measured_height(runtime_context, model_context, mode):
    positions = torch.tensor([[0., 0., .62]]).expand(3, -1)
    feet = torch.tensor([[[0., 0., .1]]]).expand(3, -1, -1)
    quaternion = torch.tensor([[1., 0., 0., 0.]]).expand(3, -1)
    def robot(origins, directions, limit, env_ids):
        return torch.tensor([torch.nan, .28, torch.nan]).reshape(3, 1, 1)
    manager, _ = _manager(runtime_context, _model(model_context, (0,)), HorizontalPlaneRayQuery(0.),
                          robot, layers=0, hit_mode=mode)
    state = _read(manager, positions, quaternion, feet)
    assert state.foot_terrain_scan_heights is not None and state.foot_terrain_scan_valid is not None
    centre = .1 if mode == "ground_only" else -.24
    torch.testing.assert_close(state.foot_terrain_scan_heights.flatten(), torch.tensor([.1, centre, .1]))
    assert state.foot_terrain_scan_valid.flatten().tolist() == [True, mode != "visible_ground", True]


def test_reference_plane_and_gravity_projection(runtime_context, model_context):
    half = pi / 12
    c, s = cos(2 * half), sin(2 * half)
    rot = torch.tensor([[1., 0., 0.], [0., c, -s], [0., s, c]])
    position = torch.tensor([[1., 2., .8]])
    quat = torch.tensor([[cos(half), sin(half), 0., 0.]])
    local_feet = torch.tensor([[[.1, .2, -.35], [.1, -.2, -.4]]])
    feet = position[:, None] + local_feet @ rot.T
    rays = []
    def query(origins, directions, limit, ids):
        rays.append((origins, directions))
        return HorizontalPlaneRayQuery(0.)(origins, directions, limit, ids)
    manager, _ = _manager(runtime_context, _model(model_context), query, layers=1)
    scan = manager.terms["foot_terrain_scan"]
    assert isinstance(scan, FootTerrainScan)
    state = _read(manager, position, quat, feet)
    origins, directions = rays[-1]
    targets = torch.zeros_like(directions)
    targets[..., :2] = local_feet[:, :, None, :2] + scan.offsets
    targets[..., 2] = -.5
    expected = torch.nn.functional.normalize(targets @ rot.T, dim=-1)
    torch.testing.assert_close(origins, position[:, None].expand(-1, 2, -1))
    torch.testing.assert_close(directions, expected)
    torch.testing.assert_close(state.foot_terrain_scan_heights, feet[..., 2, None].expand(-1, -1, 7))
    assert state.foot_terrain_scan_heights is not None
    raised = _read(manager, position, quat, feet + .1 * rot[:, 2])
    torch.testing.assert_close(rays[-1][1], directions)
    torch.testing.assert_close(raised.foot_terrain_scan_heights, state.foot_terrain_scan_heights + .1*c)
    hits = origins[:, :, None] - directions * origins[..., 2, None, None] / directions[..., 2:3]
    local_hits = (hits-position[:, None, None]) @ rot
    offset = (local_hits-targets)[..., :2].norm(dim=-1)
    tangent = targets[..., :2].norm(dim=-1) / .5
    torch.testing.assert_close(offset, (local_hits[..., 2]+.5).abs()*tangent)

    def slope(origins, directions, limit, ids):
        return -.5 / (directions @ rot[:, 2])
    manager, _ = _manager(runtime_context, _model(model_context), slope, layers=1)
    sloped = _read(manager, position, quat, feet)
    expected_height = feet[..., 2, None] - (position[:, None, None] + targets @ rot.T)[..., 2]
    torch.testing.assert_close(sloped.foot_terrain_scan_heights, expected_height)


def test_mask_invalid_pose_foot_and_view(runtime_context, model_context):
    manager, _ = _manager(runtime_context, _model(model_context), HorizontalPlaneRayQuery(0.),
                          layers=0, max_angle_deg=30.)
    feet = torch.tensor([[[0., 0., .2], [1., 0., .3]]]).repeat(3, 1, 1)
    feet[1, 0, 0] = torch.nan
    pos = torch.tensor([[0., 0., .5]]).repeat(3, 1)
    quat = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    quat[-1, 0] = torch.nan
    state = _read(manager, pos, quat, feet)
    assert state.foot_terrain_scan_heights is not None and state.foot_terrain_scan_valid is not None
    assert state.foot_terrain_scan_valid.squeeze(-1).tolist() == [[True,False],[False,False],[False,False]]
    for ids in [torch.tensor([2,0]), torch.empty(0, dtype=torch.long)]:
        subset = _read(manager, pos[ids], quat[ids], feet[ids], ids)
        torch.testing.assert_close(subset.foot_terrain_scan_heights, state.foot_terrain_scan_heights[ids])

def test_only_moving_foot_redirects_its_beams(runtime_context, model_context):
    batches = []
    def query(origins, directions, limit, ids):
        batches.append((origins.clone(), directions.clone()))
        return HorizontalPlaneRayQuery(0.)(origins, directions, limit, ids)

    manager, _ = _manager(runtime_context, _model(model_context), query, layers=1)
    position = torch.tensor([[0., 0., .6]])
    quat = torch.tensor([[1., 0., 0., 0.]])
    feet = torch.tensor([[[.2, .1, .2], [-.1, .2, .1]]])
    _read(manager, position, quat, feet)
    feet[:, 0, 0] += .13
    _read(manager, position, quat, feet)
    torch.testing.assert_close(batches[0][0], batches[1][0])
    torch.testing.assert_close(batches[0][1][:, 1], batches[1][1][:, 1])
    assert not torch.allclose(batches[0][1][:, 0], batches[1][1][:, 0])


@pytest.mark.parametrize(("mode", "expected_height", "valid"), [
    ("first_surface", -.24, True),
    ("visible_ground", -.24, False),
    ("ground_only", 0., False),
])
def test_occlusion_inside_range_with_ground_outside(runtime_context, model_context, mode, expected_height, valid):
    def robot(origins, directions, limit, env_ids):
        return torch.full_like(directions[..., 0], .28)

    manager, _ = _manager(runtime_context, _model(model_context, (0,)), HorizontalPlaneRayQuery(0.),
                          robot, layers=0, hit_mode=mode, max_distance=.5)
    state = _read(manager, torch.tensor([[0., 0., .62]]), torch.tensor([[1., 0., 0., 0.]]),
                  torch.tensor([[[0., 0., .1]]]))
    assert state.foot_terrain_scan_valid is not None and state.foot_terrain_scan_valid.item() == valid
    assert state.foot_terrain_scan_heights is not None
    assert state.foot_terrain_scan_heights.item() == pytest.approx(expected_height)


def test_optical_axis_cone_includes_edge(runtime_context, model_context):
    manager, _ = _manager(runtime_context, _model(model_context), HorizontalPlaneRayQuery(0.),
                          layers=0, max_angle_deg=45.)
    state = _read(manager, torch.tensor([[0., 0., .5]]), torch.tensor([[1., 0., 0., 0.]]),
                  torch.tensor([[[.5, 0., 0.], [.501, 0., 0.]]]))
    assert state.foot_terrain_scan_valid is not None
    assert state.foot_terrain_scan_valid.flatten().tolist() == [True, False]
