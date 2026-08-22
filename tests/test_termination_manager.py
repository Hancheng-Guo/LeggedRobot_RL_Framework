import pytest
import torch

from envs.tasks.managers.termination.base import TerminationManager
from envs.tasks.managers.termination.terms.base import BaseTerminationTerm
from envs.tasks.managers.termination.terms.registry import (
    TERMINATION_CLASS_MAP,
)
from envs.tasks.utils.context import TaskContext


def make_task_context() -> TaskContext:
    qpos = torch.zeros(2, 9)
    qpos[:, 7] = torch.tensor([0.1, 0.3])
    return TaskContext(
        state={"qpos": qpos},
        command={},
        action=torch.zeros(2, 2),
        last_action=torch.zeros(2, 2),
        episode_step=torch.zeros(2, dtype=torch.long),
    )


def test_termination_manager_combines_terms_and_reports_info(
    runtime_context,
    model_context,
):
    manager = TerminationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={"base_height": {"min_height": 0.2}},
    )

    terminated, info = manager.compute(make_task_context())

    torch.testing.assert_close(terminated, torch.tensor([True, False]))
    torch.testing.assert_close(info["termination/base_height"], terminated)
    assert manager.terms["base_height"].context is runtime_context


def test_termination_manager_rejects_non_bool_term(
    runtime_context,
    model_context,
):
    class InvalidTermination(BaseTerminationTerm):
        def compute(self, task_context: TaskContext) -> torch.Tensor:
            return torch.zeros(2)

    TERMINATION_CLASS_MAP["invalid_termination"] = InvalidTermination
    try:
        manager = TerminationManager(
            num_envs=2,
            context=runtime_context,
            model_context=model_context,
            terms={"invalid_termination": {}},
        )
        with pytest.raises(TypeError):
            manager.compute(make_task_context())
    finally:
        TERMINATION_CLASS_MAP.pop("invalid_termination", None)


def test_termination_manager_partial_reset(
    runtime_context,
    model_context,
):
    manager = TerminationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={"base_height": {"min_height": 0.2}},
    )
    manager.compute(make_task_context())

    manager.reset(torch.tensor([0]))

    assert not manager.term_terminated["base_height"][0]


def test_body_contact_terminates_for_either_contact_order(
    runtime_context,
    model_context,
):
    manager = TerminationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "body_contact": {
                "body_names": ["base", "thigh"],
                "ground_geom_name": "floor",
            },
        },
    )
    task_context = make_task_context()
    task_context.state["contact_geom_ids"] = torch.tensor([
        [[1, 0], [-1, -1]],
        [[0, 2], [-1, -1]],
    ])

    terminated, info = manager.compute(task_context)

    torch.testing.assert_close(terminated, torch.tensor([True, True]))
    torch.testing.assert_close(
        info["termination/body_contact"],
        terminated,
    )


def test_body_contact_ignores_other_contacts(
    runtime_context,
    model_context,
):
    manager = TerminationManager(
        num_envs=2,
        context=runtime_context,
        model_context=model_context,
        terms={
            "body_contact": {
                "body_names": ["base"],
                "ground_geom_name": "floor",
            },
        },
    )
    task_context = make_task_context()
    task_context.state["contact_geom_ids"] = torch.tensor([
        [[3, 0]],
        [[1, 3]],
    ])

    terminated, _ = manager.compute(task_context)

    torch.testing.assert_close(terminated, torch.tensor([False, False]))


def test_body_contact_rejects_unknown_body_name(
    runtime_context,
    model_context,
):
    with pytest.raises(ValueError, match="missing_body"):
        TerminationManager(
            num_envs=2,
            context=runtime_context,
            model_context=model_context,
            terms={
                "body_contact": {
                    "body_names": ["missing_body"],
                    "ground_geom_name": "floor",
                },
            },
        )
