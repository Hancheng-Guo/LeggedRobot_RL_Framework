"""Public lifecycle contracts that must survive app/runner refactoring."""
import pytest
import torch
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from runners.on_policy import OnPolicyRunner
from runners.types import RestoreMode, TrainStopReason
from app.utils.run_workspace import inspect_checkpoint_metadata
from envs import BaseEnv
from rl.algorithms import OnPolicyAlgorithm
from utils import Component, RuntimeContext


def test_training_layers_do_not_import_application() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = []
    for directory in ("runners", "rl", "envs"):
        for path in (root / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                modules = []
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                if any(name == "app" or name.startswith("app.") for name in modules):
                    violations.append(f"{path.relative_to(root)}:{getattr(node, 'lineno', 0)}")
    assert not violations, violations


def test_application_does_not_access_callback_implementation() -> None:
    root = Path(__file__).resolve().parents[1] / "app"
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("runners.callbacks"):
                violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr in {"_pending_callback_states", "stop_callback"}:
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, violations


def test_application_reads_checkpoint_metadata_without_runner(
    tmp_path: Path,
) -> None:
    from app.utils.run_workspace import select_checkpoint

    path = tmp_path / "checkpoints" / "resume.pt"
    path.parent.mkdir(parents=True)
    torch.save({"runner": {
        "stage_index": 1,
        "current_iteration": 42,
        "callbacks": [{"type": "StageCallback", "state": {"stop_training": True}}],
    }}, path)
    selected = select_checkpoint(
        tmp_path, 2,
    )
    assert selected.checkpoint == path
    assert selected.stage_index == 1
    assert selected.stage_completed is True


@pytest.mark.parametrize("completed", [False, True])
def test_callback_checkpoint_state_overrides_directory_marker(
    tmp_path: Path, completed: bool,
) -> None:
    path = tmp_path / "latest.pt"
    torch.save({"runner": {
        "stage_index": 0, "current_iteration": 7,
        "callbacks": [{"type": "StageCallback", "state": {
            "stop_training": completed, "condition_values": [[1.0]],
        }}],
    }}, path)
    (tmp_path / "stage_completed").touch()
    info = inspect_checkpoint_metadata(path)
    assert info.stage_completed is completed
    assert info.stage_index == 0 and info.current_iteration == 7


@pytest.mark.parametrize("mode", list(RestoreMode))
def test_restore_mode_controls_optimizer_and_callback_history(
    tmp_path: Path, runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch, mode: RestoreMode,
) -> None:
    runner = OnPolicyRunner(runtime_context)
    restored_optimizer = []
    runner.algorithm = cast(OnPolicyAlgorithm, SimpleNamespace(
        set_train_mode=lambda: None,
        load_checkpoint_state_dict=lambda state, load_optimizer: restored_optimizer.append(load_optimizer),
    ))
    runner.environment = cast(BaseEnv, SimpleNamespace(num_envs=1, reset=lambda: torch.zeros(1, 1)))
    monkeypatch.setattr(runner, "_build_environment", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_build_algorithm", lambda **kwargs: None)
    transition = [{"metric": "score", "method": "mean", "operator": ">=",
                   "threshold": 100.0, "window": 3}]
    runner.config_update(
        Component(None, None, None, None, None, None),
        max_iterations=1, rollout_length=1, stage_index=0, transition=transition,
    )
    path = tmp_path / "resume.pt"
    torch.save({"runner": {
        "stage_index": 0, "current_iteration": 0, "algorithm": {},
        "callbacks": [{"type": "StageCallback", "state": {
            "stop_training": False, "condition_values": [[7.0, 9.0]],
        }}],
    }}, path)
    runner.prepare_checkpoint_load(path)
    runner.load(mode=mode)
    assert restored_optimizer == [mode is not RestoreMode.EVALUATE]

    # Even with no iteration budget left, start hooks initialize the callback.
    # RESUME must then restore history; advance/evaluate must not restore it.
    result = runner.train()
    state = runner.callbacks[-1].checkpoint_state_dict()
    assert state["condition_values"] == ([[7.0, 9.0]] if mode is RestoreMode.RESUME else [[]])
    assert result.reason is TrainStopReason.MAX_ITERATIONS_REACHED
    assert result.current_iteration == 0


def test_manifest_completion_does_not_override_checkpoint(
    tmp_path: Path,
) -> None:
    import json
    from app.utils.run_workspace import select_checkpoint

    path = tmp_path / "checkpoints" / "resume.pt"
    path.parent.mkdir()
    torch.save({"runner": {
        "stage_index": 0, "current_iteration": 3,
        "callbacks": [{"type": "StageCallback", "state": {"stop_training": False}}],
    }}, path)
    (tmp_path / "resume_manifest.json").write_text(json.dumps({
        "status": "ready",
        "stage_index": 0, "stage_completed": True,
    }), encoding="utf-8")
    selected = select_checkpoint(
        tmp_path, 1,
    )
    assert selected.stage_completed is False


@pytest.mark.parametrize("evaluation", ["test", "play"])
def test_stage_training_after_evaluation_rebuilds_completion_callback(
    tmp_path: Path, runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch, evaluation: str,
) -> None:
    from app.stage_manager import StageManager

    context = replace(runtime_context, save_dir=tmp_path, load_dir=tmp_path)
    transition = [{"metric": "score", "method": "mean", "operator": ">=",
                   "threshold": 0.5, "window": 1}]
    runner = OnPolicyRunner(context)
    obs = torch.zeros(1, 1)
    runner.environment = cast(BaseEnv, SimpleNamespace(
        num_envs=1, reset=lambda: obs, close=lambda: None,
        has_playback_render_mode=lambda: False,
        step=lambda action: (obs, obs, torch.ones(1), torch.ones(1, dtype=torch.bool),
                             torch.zeros(1, dtype=torch.bool), {"score": torch.ones(1)}),
    ))
    runner.algorithm = cast(OnPolicyAlgorithm, SimpleNamespace(
        set_eval_mode=lambda: None, set_train_mode=lambda: None, close=lambda: None,
        act=lambda obs, deterministic=False: SimpleNamespace(action=torch.zeros(1, 1)),
        reset_policy_state=lambda *args: None,
        process_transition=lambda **kwargs: None,
        compute_returns=lambda **kwargs: None,
        update=lambda: {"score": 1.0},
        checkpoint_state_dict=lambda: {},
        save_module_artifacts=lambda directory: [],
    ))
    monkeypatch.setattr(runner, "_build_environment", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_build_algorithm", lambda **kwargs: None)
    runner.config_update(
        Component(None, None, None, None, None, None),
        max_iterations=1, rollout_length=1, stage_index=0, transition=transition,
    )
    manager = StageManager(
        component={}, context=context, load_dir=tmp_path,
        stage_detail=[{"one": {"max_iterations": 1, "transition": transition}}],
    )
    manager.runner = runner
    if evaluation == "test":
        manager.test(num_episodes=1)
    else:
        manager.play(num_steps=1, formats="gif", num_plays=1)
    manager.train()
    assert not manager.continue_training
    saved = tmp_path / "checkpoints" / "stage_000" / "latest.pt"
    assert inspect_checkpoint_metadata(saved).stage_completed is True
    assert not (saved.parent / "stage_completed").exists()
    if evaluation == "test":
        manager.test(num_episodes=1)
    else:
        manager.play(num_steps=1, formats="gif", num_plays=1)
    manager.save()
    assert inspect_checkpoint_metadata(saved).stage_completed is True
    manager.close()


@pytest.mark.parametrize("mode", list(RestoreMode))
def test_save_after_restore_preserves_completed_stage(
    tmp_path: Path, runtime_context: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch, mode: RestoreMode,
) -> None:
    callbacks = [{"type": "StageCallback", "state": {
        "stop_training": True, "condition_values": [[1.0]],
    }}]
    source = tmp_path / "source.pt"
    torch.save({"runner": {
        "stage_index": 0, "current_iteration": 4,
        "algorithm": {}, "callbacks": callbacks,
    }}, source)
    runner = OnPolicyRunner(runtime_context)
    runner.algorithm = cast(OnPolicyAlgorithm, SimpleNamespace(
        load_checkpoint_state_dict=lambda *args, **kwargs: None,
        checkpoint_state_dict=lambda: {},
        save_module_artifacts=lambda path: [],
    ))
    monkeypatch.setattr(runner, "_build_environment", lambda **kwargs: None)
    monkeypatch.setattr(runner, "_build_algorithm", lambda **kwargs: None)
    transition = [{"metric": "score", "method": "mean", "operator": ">=",
                   "threshold": 0.5, "window": 1}]
    runner.prepare_checkpoint_load(source)
    runner.config_update(
        Component(None, None, None, None, None, None),
        max_iterations=5, rollout_length=1, stage_index=0,
        transition=None if mode is RestoreMode.EVALUATE else transition,
    )
    runner.load(mode=mode)
    saved = runner.save(tmp_path / "resaved.pt")
    assert inspect_checkpoint_metadata(saved).stage_completed is True
    assert torch.load(saved, weights_only=False)["runner"]["callbacks"] == callbacks
    if mode is not RestoreMode.RESUME:
        assert runner._pending_callback_states is None
