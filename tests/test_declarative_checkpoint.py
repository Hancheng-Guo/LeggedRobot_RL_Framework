"""Persistent state declarations compose independently from reset-only buffers."""
from copy import deepcopy

import pytest
import torch

from envs.tasks.utils import (
    ResetStateOwner, StateOwner, StateManager,
    ResetSnapshot, capture_task_reset_state,
)
from utils import CheckpointStateOwner, CheckpointStateManager
from test_terrain_curriculum import make_task, reset


MANAGERS = (
    'action_manager', 'command_manager', 'observation_manager', 'reward_manager',
    'termination_manager', 'task_state_manager', 'curriculum_manager', 'randomization_manager',
)


@pytest.mark.parametrize('name', MANAGERS)
def test_all_managers_and_terms_share_combined_state_capabilities(runtime_context, model_context, name):
    task, backend = make_task(runtime_context, model_context)
    manager = getattr(task, name)
    assert isinstance(manager, StateManager)
    snapshot = manager.capture_reset_state()
    assert type(snapshot) is ResetSnapshot
    snapshot.restore()
    for term in manager.terms.values():
        assert isinstance(term, StateOwner)
        term_snapshot = term.capture_reset_state()
        assert type(term_snapshot) is ResetSnapshot
        term_snapshot.restore()


def test_task_snapshot_has_the_same_type_and_flat_captured_data(runtime_context, model_context):
    task, backend = make_task(runtime_context, model_context)
    snapshot = capture_task_reset_state([task.curriculum_manager, task.command_manager])
    assert type(snapshot) is ResetSnapshot
    assert snapshot.tensors and snapshot.generators
    assert not hasattr(snapshot, 'states')
    snapshot.restore()


@pytest.mark.parametrize('name', MANAGERS)
def test_all_managers_support_declared_checkpoint_fields(runtime_context, model_context, monkeypatch, name):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    manager = getattr(task, name)
    monkeypatch.setattr(type(manager), 'CHECKPOINT_STATE_FIELDS', ('_test_counter',))
    setattr(manager, '_test_counter', torch.tensor([3, 7]))
    state = manager.checkpoint_state_dict()
    assert state['_test_counter'].device.type == 'cpu'
    manager._test_counter.zero_()
    torch.testing.assert_close(state['_test_counter'], torch.tensor([3, 7]))
    manager.load_checkpoint_state_dict(state)
    torch.testing.assert_close(manager._test_counter, torch.tensor([3, 7]))


def test_shared_and_purpose_specific_fields():
    class Owner(ResetStateOwner, CheckpointStateOwner):
        STATE_FIELDS = ('shared',)
        RESET_STATE_FIELDS = ('episode',)
        CHECKPOINT_STATE_FIELDS = ('total',)
        def __init__(self):
            self.shared = torch.tensor([1.])
            self.episode = torch.tensor([2.])
            self.total = torch.tensor([3.])
    owner = Owner()
    checkpoint = owner.checkpoint_state_dict()
    rollback = owner.capture_reset_state()
    assert set(checkpoint) == {'shared', 'total'}
    owner.shared.zero_()
    owner.episode.zero_()
    owner.total.zero_()
    rollback.restore()
    assert owner.shared.item() == 1 and owner.episode.item() == 2 and owner.total.item() == 0
    owner.episode.fill_(9)
    owner.load_checkpoint_state_dict(checkpoint)
    assert owner.total.item() == 3 and owner.episode.item() == 9


def test_entire_manager_is_validated_before_any_mutation():
    class Term(CheckpointStateOwner):
        CHECKPOINT_STATE_FIELDS = ('weight',)
        def __init__(self):
            self.weight = torch.ones(2)
    class Manager(CheckpointStateManager):
        CHECKPOINT_STATE_FIELDS = ('counter',)
        def __init__(self):
            self.counter = 2
            self.terms = {'first': Term(), 'second': Term()}
    manager = Manager()
    state = deepcopy(manager.checkpoint_state_dict())
    state['counter'] = 99
    state['terms']['first']['weight'].zero_()
    state['terms']['second']['weight'] = torch.zeros(3)
    with pytest.raises(ValueError, match='tensor'):
        manager.load_checkpoint_state_dict(state)
    assert manager.counter == 2
    torch.testing.assert_close(manager.terms['first'].weight, torch.ones(2))


def test_base_task_collects_declared_common_manager_state(runtime_context, model_context, monkeypatch):
    task, backend = make_task(runtime_context, model_context)
    reset(task, backend)
    manager = task.action_manager
    monkeypatch.setattr(type(manager), 'CHECKPOINT_STATE_FIELDS', ('_test_total',))
    setattr(manager, '_test_total', 13)
    state = task.checkpoint_state_dict()
    assert state['action_manager'] == {'_test_total': 13}
    setattr(manager, '_test_total', 0)
    task.load_checkpoint_state_dict(state)
    assert getattr(manager, '_test_total') == 13


def test_checkpoint_metadata_is_validated():
    class Owner(CheckpointStateOwner):
        def _checkpoint_metadata(self):
            return {'version': 1}
    with pytest.raises(ValueError, match='metadata'):
        Owner().load_checkpoint_state_dict({'version': 2})


def test_shared_alias_fields_use_the_same_declaration_for_both_purposes():
    class Owner(ResetStateOwner, CheckpointStateOwner):
        STATE_FIELDS = {'statistics.total': 'total'}
        def __init__(self):
            self.total = torch.tensor([5.])
    owner = Owner()
    checkpoint = owner.checkpoint_state_dict()
    rollback = owner.capture_reset_state()
    owner.total.zero_()
    rollback.restore()
    assert owner.total.item() == 5
    owner.total.zero_()
    owner.load_checkpoint_state_dict(checkpoint)
    assert owner.total.item() == 5


def test_term_load_guards_are_checked_before_manager_mutation():
    class Term(CheckpointStateOwner):
        def _before_checkpoint_load(self):
            raise RuntimeError('pending transaction')
    class Manager(CheckpointStateManager):
        CHECKPOINT_STATE_FIELDS = ('counter',)
        CHECKPOINT_KEEP_EMPTY_TERMS = True
        def __init__(self):
            self.counter = 2
            self.terms = {'term': Term()}
    manager = Manager()
    state = manager.checkpoint_state_dict()
    state['counter'] = 99
    with pytest.raises(RuntimeError, match='pending transaction'):
        manager.load_checkpoint_state_dict(state)
    assert manager.counter == 2
