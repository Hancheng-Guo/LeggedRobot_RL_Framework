"""Declarative checkpoint state shared by task and simulator managers."""
import torch
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, ClassVar


def state_field_paths(
    owner: Any,
    *declarations: str,
) -> dict[str, str]:

    result: dict[str, str] = {}
    for cls in reversed(type(owner).__mro__):
        for declaration in declarations:
            fields = cls.__dict__.get(declaration, ())
            result.update(fields if isinstance(fields, Mapping) else {name: name for name in fields})
    return result


def checkpoint_value(value: Any) -> Any:

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, torch.Generator):
        return value.get_state().cpu().clone()
    if isinstance(value, Mapping):
        return {key: checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [checkpoint_value(item) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, set):
        return [checkpoint_value(item) for item in sorted(value)]
    return deepcopy(value)


def _read_attribute(
    owner: Any,
    path: str,
) -> Any:

    for name in path.split('.'):
        owner = getattr(owner, name)
    return owner


def _read_state(
    state: Mapping[str, Any],
    path: str,
) -> Any:

    value: Any = state
    for name in path.split('.'):
        if not isinstance(value, Mapping) or name not in value:
            raise ValueError(f"Missing checkpoint field: {path}.")
        value = value[name]
    return value


def _write_state(
    state: dict[str, Any],
    path: str,
    value: Any,
) -> None:

    names = path.split('.')
    for name in names[:-1]:
        state = state.setdefault(name, {})
    state[names[-1]] = value


def _validate_structure(
    saved: Any,
    current: Any,
    path: str,
) -> None:

    if isinstance(current, Mapping):
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            raise ValueError(f"Checkpoint fields do not match: {path}.")
        for key, value in current.items():
            _validate_structure(saved[key], value, f'{path}.{key}')
    elif isinstance(current, torch.Tensor):
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError(f"Invalid checkpoint tensor: {path}.")
    elif type(saved) is not type(current):
        raise ValueError(f"Invalid checkpoint field type: {path}.")
    elif isinstance(current, (list, tuple)):
        if len(saved) != len(current):
            raise ValueError(f"Invalid checkpoint sequence length: {path}.")
        for index, value in enumerate(current):
            _validate_structure(saved[index], value, f'{path}[{index}]')


def _validate_metadata(
    saved: Any,
    expected: Any,
    path: str,
) -> None:

    if isinstance(expected, Mapping):
        if not isinstance(saved, Mapping) or not set(expected).issubset(saved):
            raise ValueError(f"Checkpoint metadata does not match: {path}.")
        for name, value in expected.items():
            _validate_metadata(saved[name], value, f'{path}.{name}')
    elif isinstance(expected, torch.Tensor):
        if not isinstance(saved, torch.Tensor) or not torch.equal(saved.cpu(), expected.cpu()):
            raise ValueError(f"Checkpoint metadata does not match: {path}.")
    elif saved != expected:
        raise ValueError(f"Checkpoint metadata does not match: {path}.")


class CheckpointStateOwner:
    # STATE_FIELDS participate in both checkpoint and reset; each purpose may add fields.

    STATE_FIELDS: ClassVar[tuple[str, ...] | Mapping[str, str]] = ()
    CHECKPOINT_STATE_FIELDS: ClassVar[tuple[str, ...] | Mapping[str, str]] = ()


    def _checkpoint_fields(self) -> dict[str, str]:
        return state_field_paths(self, 'STATE_FIELDS', 'CHECKPOINT_STATE_FIELDS')


    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {}


    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = checkpoint_value(self._checkpoint_metadata())
        for key, path in self._checkpoint_fields().items():
            _write_state(state, key, checkpoint_value(_read_attribute(self, path)))
        return state


    def validate_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Checkpoint state must be a mapping.")
        expected = CheckpointStateOwner.checkpoint_state_dict(self)
        _validate_structure(state, expected, type(self).__name__)
        _validate_metadata(state, self._checkpoint_metadata(), type(self).__name__)


    def _restore_checkpoint_fields(
        self,
        state: Mapping[str, Any],
    ) -> None:
        with torch.no_grad():
            for key, path in self._checkpoint_fields().items():
                value = _read_state(state, key)
                names = path.split('.')
                owner = self
                for name in names[:-1]:
                    owner = getattr(owner, name)
                target = getattr(owner, names[-1])
                if isinstance(target, torch.Tensor):
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
                elif isinstance(target, torch.Generator):
                    target.set_state(value.cpu())
                elif isinstance(target, set):
                    target.clear()
                    target.update(value)
                else:
                    setattr(owner, names[-1], deepcopy(value))


    def _before_checkpoint_load(self) -> None:
        pass


    def _after_checkpoint_load(
        self,
        state: Mapping[str, Any],
    ) -> None:
        pass


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self._before_checkpoint_load()
        self.validate_checkpoint_state_dict(state)
        self._restore_checkpoint_fields(state)
        self._after_checkpoint_load(state)


    def playback_state_dict(
        self,
        env_index: int,
    ) -> dict[str, Any]:
        state = self.checkpoint_state_dict()
        if state:
            raise NotImplementedError(
                "Stateful owners must implement playback_state_dict()."
            )
        return state


class CheckpointStateManager(CheckpointStateOwner):

    terms: Any
    num_envs: int
    CHECKPOINT_KEEP_EMPTY_TERMS: ClassVar[bool] = False


    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = super().checkpoint_state_dict()
        terms = {
            name: term.checkpoint_state_dict()
            for name, term in self.terms.items()
        }
        if self.CHECKPOINT_KEEP_EMPTY_TERMS or any(terms.values()):
            state['terms'] = terms
        return state


    def validate_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if not isinstance(state, Mapping):
            raise TypeError(
                f"{type(self).__name__} checkpoint must be a mapping."
            )
        own = {key: value for key, value in state.items() if key != 'terms'}
        CheckpointStateOwner.validate_checkpoint_state_dict(self, own)
        terms = state.get('terms')
        needed = self.CHECKPOINT_KEEP_EMPTY_TERMS or any(
            term.checkpoint_state_dict() for term in self.terms.values())
        if not needed and terms is None:
            return
        if not isinstance(terms, Mapping) or set(terms) != set(self.terms):
            raise ValueError(
                f"Checkpoint {type(self).__name__} terms must match configured terms."
            )
        for name, term in self.terms.items():
            term._before_checkpoint_load()
            term.validate_checkpoint_state_dict(terms[name])


    def load_checkpoint_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self._before_checkpoint_load()
        self.validate_checkpoint_state_dict(state)
        self._restore_checkpoint_fields(state)
        for name, saved in state.get('terms', {}).items():
            self.terms[name].load_checkpoint_state_dict(saved)
        self._after_checkpoint_load(state)


    def playback_state_dict(
        self,
        env_index: int,
    ) -> dict[str, Any]:
        if not 0 <= env_index < self.num_envs:
            raise ValueError("Playback environment index is out of range.")
        state = (
            CheckpointStateOwner.playback_state_dict(self, env_index)
            if self._checkpoint_fields() else {}
        )
        terms = {name: term.playback_state_dict(env_index) for name, term in self.terms.items()}
        if self.CHECKPOINT_KEEP_EMPTY_TERMS or any(terms.values()):
            state['terms'] = terms
        return state
