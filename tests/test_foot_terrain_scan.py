from dataclasses import replace
from math import cos, pi, sin, sqrt
from types import SimpleNamespace

import pytest
import torch

from envs.simulators.state.terms.foot_terrain_scan import FootTerrainScan
from envs.simulators.utils.ground_ray import GROUND_RAY_QUERY, HorizontalGroundRayQuery
from envs.simulators.state.terms import BaseSimulatorStateTerm, register_simulator_state_term
from envs.simulators.state.terms import registry
from envs.simulators.state import SimulatorStateManager


def _manager(context, model_context, query, **config):
    manager = SimulatorStateManager(
        context=context,
        model_context=model_context,
        simulator=SimpleNamespace(query_ground_rays=query),
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
    manager, model = _manager(runtime_context, _model(model_context), HorizontalGroundRayQuery(0.))
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
    {"max_distance": float("inf")}, {"max_distance": -1},
])
def test_scan_config_validation(runtime_context, model_context, params):
    with pytest.raises(ValueError, match="foot_terrain_scan"):
        _manager(runtime_context, model_context, HorizontalGroundRayQuery(0.), **params)


def test_scan_rejects_unknown_config_parameter(runtime_context, model_context):
    with pytest.raises(TypeError, match="origin_height"):
        _manager(runtime_context, model_context, HorizontalGroundRayQuery(0.), origin_height=.6)


def test_disabled_manager_preserves_existing_fields(runtime_context, model_context):
    model = _model(model_context)
    calls = []

    def query(origins, directions, max_distance, env_ids):
        calls.append(origins.shape)
        return HorizontalGroundRayQuery(0.)(origins, directions, max_distance, env_ids)

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
    assert calls == [torch.Size([1, 2, 61, 3])]
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


def test_plane_ray_query_includes_range_endpoints_and_rejects_parallel_rays():
    origins = torch.zeros((2, 1, 4, 3))
    origins[0, 0, :, 2] = torch.tensor([.5, 2.5, 2.51, .49])
    directions = torch.tensor([[0., 0., -1.], [1., 0., 0.]])
    distance = HorizontalGroundRayQuery(.5)(
        origins=origins, directions=directions, max_distance=2., env_ids=None,
    )
    torch.testing.assert_close(distance[0, 0, :2], torch.tensor([0., 2.]))
    assert distance[0, 0, 2:].isnan().all()
    assert distance[1].isnan().all()


def test_scan_uses_the_current_simulator_query(runtime_context, model_context):
    simulator = SimpleNamespace(query_ground_rays=HorizontalGroundRayQuery(0.))
    manager = SimulatorStateManager(
        runtime_context, _model(model_context),
        terms={"foot_terrain_scan": {"layers": 0}}, simulator=simulator,
    )
    position = torch.tensor([[0., 0., .5]])
    quaternion = torch.tensor([[1., 0., 0., 0.]])
    feet = torch.tensor([[[0., .2, .2], [0., -.2, .2]]])

    before = _read(manager, position, quaternion, feet)
    simulator.query_ground_rays = HorizontalGroundRayQuery(.1)
    after = _read(manager, position, quaternion, feet)

    torch.testing.assert_close(before.foot_terrain_scan_heights, torch.full((1, 2, 1), -.2))
    torch.testing.assert_close(after.foot_terrain_scan_heights, torch.full((1, 2, 1), -.1))


def test_tilted_plane_queries_keep_full_pose_and_gravity_projection(runtime_context, model_context):
    positions = torch.tensor([[1., 2., .8]]).expand(3, -1)
    feet = torch.tensor([[[1.1, 2., .2], [1., 2.1, .4]]]).expand(3, -1, -1)
    half = pi / 12
    quats = torch.tensor([[cos(half), sin(half), 0., 0.],
                          [cos(half), 0., sin(half), 0.],
                          [cos(half), 0., 0., sin(half)]])
    c, s = cos(2 * half), sin(2 * half)
    rotations = torch.tensor([[[1., 0., 0.], [0., c, -s], [0., s, c]],
                              [[c, 0., s], [0., 1., 0.], [-s, 0., c]],
                              [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]])
    captured = []

    def query(origins, directions, max_distance, env_ids):
        distance = HorizontalGroundRayQuery(0.)(origins, directions, max_distance, env_ids)
        captured.append((origins, directions, distance))
        return distance

    manager, _ = _manager(runtime_context, _model(model_context), query, layers=1)
    state = _read(manager, positions, quats, feet)
    scan = manager.terms["foot_terrain_scan"]
    assert isinstance(scan, FootTerrainScan)
    local_feet = torch.einsum("nji,nfj->nfi", rotations, feet - positions[:, None])
    local_origins = local_feet[:, :, None, :2] + scan.offsets
    local_origins = torch.cat((local_origins, torch.zeros_like(local_origins[..., :1])), dim=-1)
    expected_origins = positions[:, None, None] + torch.einsum("nij,nfkj->nfki", rotations, local_origins)
    origins, directions, distance = captured[0]
    torch.testing.assert_close(origins, expected_origins)
    torch.testing.assert_close(directions, -rotations[..., 2])
    torch.testing.assert_close(distance, -expected_origins[..., 2] / directions[:, None, None, 2])
    valid = state.foot_terrain_scan_valid
    assert valid is not None and valid.all()
    torch.testing.assert_close(state.term_values["foot_terrain_scan_heights"], -feet[..., 2, None].expand(-1, -1, 7))

    # Set a second plane parallel to each base, one metre below it.
    def sloped_query(origins, directions, max_distance, env_ids):
        up = rotations[..., 2]
        plane = positions - up
        return (((plane[:, None, None] - origins) * up[:, None, None]).sum(dim=-1)
                / (directions * up).sum(dim=-1)[:, None, None])

    manager, _ = _manager(runtime_context, _model(model_context), sloped_query, layers=1)
    state = _read(manager, positions, quats, feet)
    hit_z = expected_origins[..., 2] - rotations[:, None, None, 2, 2]
    heights = state.term_values["foot_terrain_scan_heights"]
    assert heights is not None
    torch.testing.assert_close(heights, hit_z - feet[..., 2, None])
    assert heights[0, 0].unique().numel() > 1


def test_ray_origin_range_validity_and_foot_order(runtime_context, model_context):
    recorded = []

    def query(origins, directions, max_distance, env_ids):
        assert origins.isfinite().all() and directions.isfinite().all()
        recorded.append((origins, env_ids))
        return HorizontalGroundRayQuery(0.)(origins, directions, max_distance, env_ids)

    manager, model = _manager(runtime_context, _model(model_context, (2, 0, 1)), query, layers=0, max_distance=2.)
    assert model.state_term_output_shapes["foot_terrain_scan"]["foot_terrain_scan_heights"] == (3, 1)
    z = torch.tensor([2., 2.01, -.01, .5, .5])
    position = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), dim=-1)
    quaternion = torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1)
    quaternion[-1, 0] = torch.nan
    feet = torch.tensor([[[1., 0., .1], [2., 0., .2], [3., 0., .3]]]).repeat(5, 1, 1)
    feet[3, 1, 2] = torch.inf
    state = _read(manager, position, quaternion, feet)
    expected_valid = torch.tensor([[1,1,1],[0,0,0],[0,0,0],[1,1,0],[0,0,0]], dtype=torch.bool)[..., None]
    valid = state.foot_terrain_scan_valid
    heights = state.foot_terrain_scan_heights
    assert valid is not None and heights is not None
    assert torch.equal(valid, expected_valid)
    torch.testing.assert_close(heights[0, :, 0], torch.tensor([-.3, -.1, -.2]))
    assert torch.all(heights[~expected_valid] == 0.)
    env_ids = torch.tensor([3, 0])
    subset = _read(manager, position[env_ids], quaternion[env_ids], feet[env_ids], env_ids)
    assert recorded[-1][1] is env_ids
    selected = state.clone().select(env_ids)
    torch.testing.assert_close(subset.foot_terrain_scan_heights, selected.foot_terrain_scan_heights)
    subset_valid = subset.foot_terrain_scan_valid
    assert subset_valid is not None and torch.equal(subset_valid, expected_valid[env_ids])
    reordered, _ = _manager(runtime_context, _model(model_context, (0, 1, 2)), query, layers=0)
    other = _read(reordered, position, quaternion, feet)
    other_heights = other.foot_terrain_scan_heights
    other_valid = other.foot_terrain_scan_valid
    assert other_heights is not None and other_valid is not None
    torch.testing.assert_close(other_heights, heights[:, [1, 2, 0]])
    assert torch.equal(other_valid, expected_valid[:, [1, 2, 0]])
    before = recorded[-1][0]
    feet = torch.where(feet.isfinite(), feet + torch.tensor([0., 0., .25]), feet)
    after = _read(reordered, position, quaternion, feet)
    torch.testing.assert_close(recorded[-1][0], before)

    torch.testing.assert_close(
        after.foot_terrain_scan_heights, torch.where(other_valid, other_heights - .25, 0.),
    )
    count = len(recorded)
    empty = _read(manager, position[:0], quaternion[:0], feet[:0], torch.empty(0, dtype=torch.long))
    empty_heights = empty.foot_terrain_scan_heights
    assert empty_heights is not None and empty_heights.shape == (0, 3, 1)
    assert len(recorded) == count
