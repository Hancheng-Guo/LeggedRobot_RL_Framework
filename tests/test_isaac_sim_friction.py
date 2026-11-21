import pytest

from envs.simulators.isaac_sim_friction import _compensate, _FootMaterial, _Surface


pytestmark = pytest.mark.isaacsim


def test_compensates_average_against_the_shared_floor() -> None:
    foot = _Surface(.8, .6, "average")
    floors = {_Surface(.5, .5, "average")}
    assert _compensate(foot, floors, "multiply") == pytest.approx((1.3, 1.1))


def test_multiply_matches_different_floor_coefficients() -> None:
    foot = _Surface(.8, .6, "multiply")
    floors = {_Surface(.5, .5, "average"), _Surface(.6, .4, "min")}
    assert _compensate(foot, floors, "multiply") == (.8, .6)


@pytest.mark.parametrize("factor", [0., .8, 1., 1.2])
def test_original_multiply_scales_the_effective_contact_once(factor: float) -> None:
    foot = _Surface(.8, .6, "multiply")
    floor = _Surface(.5, .4, "average")
    static, dynamic = _compensate(foot, {floor}, "multiply")
    material = _FootMaterial(None, None, static, dynamic)
    scaled_static, scaled_dynamic = material.coefficients(factor)
    assert scaled_static * floor.static == pytest.approx(foot.static * floor.static * factor)
    assert scaled_dynamic * floor.dynamic == pytest.approx(foot.dynamic * floor.dynamic * factor)


def test_zero_friction_floor_can_use_max() -> None:
    foot = _Surface(.8, .6, "average")
    floors = {_Surface(0., 0., "average")}
    assert _compensate(foot, floors, "max") == (.4, .3)


def test_rejects_floors_that_need_different_foot_coefficients() -> None:
    foot = _Surface(.8, .6, "average")
    floors = {_Surface(.5, .5, "average"), _Surface(.6, .4, "average")}
    with pytest.raises(ValueError, match="different compensation"):
        _compensate(foot, floors, "multiply")
