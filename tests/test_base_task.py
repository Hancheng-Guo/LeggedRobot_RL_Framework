"""Default task playback must not forward unprojected multi-environment state."""
from typing import Any

import pytest

from envs.tasks.base import BaseTaskLogic


def test_default_playback_supports_empty_task_state(runtime_context):
    task = BaseTaskLogic(runtime_context)
    assert task.playback_state_dict(0) == {}


def test_default_playback_rejects_nonempty_checkpoint_state(runtime_context):
    class StatefulTask(BaseTaskLogic):
        def checkpoint_state_dict(self) -> dict[str, Any]:
            return {"assignments": [10, 20, 30]}

    task = StatefulTask(runtime_context)
    with pytest.raises(NotImplementedError, match="must implement playback_state_dict"):
        task.playback_state_dict(1)
    assert task.checkpoint_state_dict() == {"assignments": [10, 20, 30]}


def test_stateful_task_can_define_its_own_playback_projection(runtime_context):
    class StatefulTask(BaseTaskLogic):
        def checkpoint_state_dict(self) -> dict[str, Any]:
            return {"assignments": [10, 20, 30]}

        def playback_state_dict(self, env_index: int) -> dict[str, Any]:
            state = self.checkpoint_state_dict()
            return {"assignments": [state["assignments"][env_index]]}

    task = StatefulTask(runtime_context)
    assert task.playback_state_dict(1) == {"assignments": [20]}
    assert task.checkpoint_state_dict() == {"assignments": [10, 20, 30]}
