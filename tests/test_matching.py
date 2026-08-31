import pytest

from utils.matching import resolve_metric_name


def test_resolve_metric_name_requires_match_by_default() -> None:
    with pytest.raises(KeyError, match="has no metric matching"):
        resolve_metric_name(
            info={"rollout/mean_len": 10.0},
            pattern="*/missing",
            owner="Test",
        )


def test_resolve_metric_name_can_allow_no_match() -> None:
    metric_name = resolve_metric_name(
        info={"rollout/mean_len": 10.0},
        pattern="*/missing",
        owner="Test",
        require_match=False,
    )

    assert metric_name is None


def test_resolve_metric_name_rejects_ambiguous_match() -> None:
    with pytest.raises(ValueError, match="is ambiguous"):
        resolve_metric_name(
            info={"rollout/loss": 1.0, "policy/loss": 2.0},
            pattern="*/loss",
            owner="Test",
            require_match=False,
        )
