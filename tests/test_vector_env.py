import pytest
import torch
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from envs import VectorEnv
from envs.simulators import BaseSimulator
from envs.simulators.utils import SimulatorState
from envs.tasks import BaseTaskLogic
from envs.tasks.utils import TaskContext, TaskStepResult
from utils import Component, ComponentInfo


class FakeSimulator:

    SUPPORTS_CONCURRENT_INSTANCES = True

    def __init__(self, num_envs: int) -> None:
        self.state = torch.zeros(num_envs, 1)
        self.sim_dt = 0.01
        self.frame_skip = 2

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.frame_skip

    def step(self, control: torch.Tensor) -> None:
        self.state += 1.0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.state.zero_()
        else:
            self.state[env_ids] = 0.0

    def get_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> SimulatorState:
        state = self.state if env_ids is None else self.state[env_ids]
        state = state.clone()
        num_envs = state.shape[0]
        return SimulatorState(
            qpos=state,
            qvel=torch.empty(num_envs, 0),
            qacc=torch.empty(num_envs, 0),
            ctrl=torch.empty(num_envs, 0),
            geom_xpos=torch.empty(num_envs, 0, 3),
            geom_xvel=torch.empty(num_envs, 0, 6),
            actuator_force=torch.empty(num_envs, 0),
            base_lin_vel_body=torch.empty(num_envs, 3),
            base_ang_vel_body=torch.empty(num_envs, 3),
            contact_geom_ids=torch.empty(num_envs, 0, 2, dtype=torch.long),
            contact_forces=torch.empty(num_envs, 0, 6),
            foot_ground_contact=torch.empty(
                num_envs, 0, dtype=torch.bool
            ),
            foot_contact_normal_force=torch.empty(num_envs, 0),
        )


class FakeTask:

    def __init__(self, num_envs: int) -> None:
        self.command = torch.zeros(num_envs, 1)
        self.action = torch.zeros(num_envs, 1)
        self.last_action = torch.zeros(num_envs, 1)
        self.action_manager = SimpleNamespace(input_dim=1)
        self.observation_manager = SimpleNamespace(output_dim=3)
        self.last_step_result: TaskStepResult | None = None
        self.last_state_command: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.command.zero_()
            self.action.zero_()
            self.last_action.zero_()
        else:
            self.command[env_ids] = 0.0
            self.action[env_ids] = 0.0
            self.last_action[env_ids] = 0.0

    def before_simulator_reset(self, env_ids=None):
        pass

    def after_simulator_reset(self, env_ids=None):
        pass

    def build_task_context(
        self,
        state: SimulatorState,
        episode_step: torch.Tensor,
        step_dt: float,
        env_ids: torch.Tensor | None = None,
    ) -> TaskContext:
        if env_ids is None:
            command = self.command
            action = self.action
            last_action = self.last_action
        else:
            command = self.command[env_ids]
            action = self.action[env_ids]
            last_action = self.last_action[env_ids]

        return TaskContext(
            state=state,
            command={"target": command},
            last_command={"target": command.clone()},
            action=action,
            last_action=last_action,
            episode_step=episode_step,
            step_dt=step_dt,
        )

    def pre_step(self) -> dict:
        return {}

    def collect_metrics(self) -> dict:
        return {}

    def update_task_state(self, task_context: TaskContext) -> None:
        self.last_state_command = task_context.command["target"].clone()

    def process_action(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        self.last_action.copy_(self.action)
        self.action.copy_(action)
        return action, {}

    def compute_reward(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:
        reward = task_context.command["target"].squeeze(-1).clone()
        return reward, {
            "reward": reward.mean(),
            "reward/test": reward.clone(),
        }

    def check_terminated(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:
        return torch.zeros(
            task_context.action.shape[0],
            dtype=torch.bool,
        ), {}

    def post_step(
        self,
        task_context: TaskContext,
        step_result: TaskStepResult,
    ) -> dict:
        self.last_step_result = step_result
        self.command += 1.0
        return {}

    def compute_observation(
        self,
        task_context: TaskContext,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return torch.cat(
            (
                task_context.state.qpos,
                task_context.command["target"],
                task_context.action,
            ),
            dim=-1,
        ), {}

    @property
    def observation_dim(self) -> int:
        return self.observation_manager.output_dim


def make_env(max_episode_steps: int = 10) -> VectorEnv:
    env = VectorEnv.__new__(VectorEnv)
    env._playback_environment = None
    env._component = None
    env.num_envs = 2
    env.max_episode_steps = max_episode_steps
    env.current_episode_steps = torch.zeros(2, dtype=torch.long)
    env.simulator = cast(BaseSimulator, FakeSimulator(2))
    env.task = cast(BaseTaskLogic, FakeTask(2))
    return env


def test_environment_keeps_merged_component_for_playback(
    monkeypatch: pytest.MonkeyPatch,
    runtime_context,
) -> None:
    env = make_env()
    env.context = runtime_context
    monkeypatch.setattr(env.simulator, "model_context", None, raising=False)
    monkeypatch.setattr(env, "_build_simulator", lambda component: None)
    monkeypatch.setattr(env, "_build_task", lambda **kwargs: None)
    monkeypatch.setattr(env, "_prepare_task", lambda component: None)
    monkeypatch.setattr(env.task, "bind", lambda simulator: None, raising=False)
    environment_info = ComponentInfo("vector", Path("environment.yaml"))
    simulator_info = ComponentInfo("mujoco", Path("simulator.yaml"))
    task_info = ComponentInfo("task", Path("task.yaml"))
    first = Component(None, None, None, environment_info, simulator_info, task_info)
    updated_task = ComponentInfo("task", Path("updated-task.yaml"))
    second = Component(None, None, None, None, None, updated_task)

    env.config_update(first)
    env.config_update(second)

    assert env._component == Component(
        None, None, None, environment_info, simulator_info, updated_task
    )


def test_prepare_task_reuses_instance_and_config_before_model_setup(
    monkeypatch, runtime_context, model_context,
):
    events = []
    reads = []

    class PreparedTask(BaseTaskLogic):
        def __init__(self, context):
            super().__init__(context)
            events.append("create")

        def prepare(self, config):
            events.append("prepare")
            super().prepare(config)
            config["nested"]["value"] = -1

        def config_update(self, component, num_envs, model_context, **config):
            events.append("configure")
            assert config["nested"]["value"] == 7
            config["nested"]["value"] = -2

        def bind(self, simulator):
            events.append("bind")

    def load_config(path):
        reads.append(path)
        return {"nested": {"value": 7}}

    env = VectorEnv(runtime_context)
    monkeypatch.setattr("envs.vector_env.TASK_TYPE_MAP", {"prepared": PreparedTask})
    monkeypatch.setattr("envs.vector_env.load_yaml", load_config)

    def build_simulator(component):
        events.append("simulator")
        env.simulator = cast(BaseSimulator, SimpleNamespace(model_context=model_context))

    monkeypatch.setattr(env, "_build_simulator", build_simulator)
    task_info = ComponentInfo("prepared", Path("task.yaml"))
    env.config_update(Component(None, None, None, None, None, task_info), num_envs=2, max_episode_steps=10)
    original_task = env.task
    env.config_update(Component(None, None, None, None, None, None))
    assert env.task is original_task
    assert reads == [Path("task.yaml")]
    assert events == ["create", "prepare", "simulator", "configure", "bind",
                      "prepare", "simulator", "configure", "bind"]


@pytest.mark.parametrize("task_is_none", [False, True], ids=["missing", "none"])
@pytest.mark.parametrize("explicit_config", [False, True], ids=["inherit", "explicit"])
def test_prepare_task_handles_absent_instance(
    runtime_context, monkeypatch, task_is_none, explicit_config,
):
    env = VectorEnv(runtime_context)
    env.num_envs = 2
    if task_is_none:
        env.task = cast(BaseTaskLogic, None)
    task_info = ComponentInfo("base", Path("task.yaml")) if explicit_config else None
    component = Component(None, None, None, None, None, task_info)
    if explicit_config:
        monkeypatch.setattr("envs.vector_env.TASK_TYPE_MAP", {"base": BaseTaskLogic})
        monkeypatch.setattr("envs.vector_env.load_yaml", lambda path: {})
        env._prepare_task(component)
        assert isinstance(env.task, BaseTaskLogic)
        assert not hasattr(env.task, "randomization_manager")
        assert env.task.get_simulator_requirements() == {}
    else:
        with pytest.raises(RuntimeError, match="task instance is required"):
            env._prepare_task(component)


def test_simulator_build_consumes_task_requirements_without_manager_access(runtime_context):
    env = VectorEnv(runtime_context)
    env.num_envs = 2
    calls = []
    # This task deliberately exposes no randomization_manager attribute.
    env.task = cast(BaseTaskLogic, SimpleNamespace(
        get_simulator_requirements=lambda: {"custom_resource": 7},
    ))
    env.simulator = cast(BaseSimulator, SimpleNamespace(
        config_update=lambda **kwargs: calls.append(kwargs),
    ))
    component = Component(None, None, None, None, None, None)
    env._build_simulator(component)
    assert calls == [{"component": component, "num_envs": 2, "custom_resource": 7}]


@pytest.mark.parametrize("env_ids", [None, torch.tensor([1])], ids=["full", "partial"])
def test_environment_owns_physics_reset_between_task_hooks(monkeypatch, env_ids):
    env = make_env()
    events = []
    original_reset = env.simulator.reset

    def check_ids(ids):
        if env_ids is None:
            assert ids is None
        else:
            torch.testing.assert_close(ids, env_ids)

    def before_simulator_reset(ids=None):
        check_ids(ids)
        events.append("before")

    def reset(ids=None):
        check_ids(ids)
        events.append("physics")
        original_reset(ids)

    def after_simulator_reset(ids=None):
        check_ids(ids)
        events.append("after")

    monkeypatch.setattr(env.task, "before_simulator_reset", before_simulator_reset)
    monkeypatch.setattr(env.simulator, "reset", reset)
    monkeypatch.setattr(env.task, "after_simulator_reset", after_simulator_reset)
    if env_ids is None:
        env.reset()
    else:
        env._reset_done_envs(torch.zeros(2, 3), torch.tensor([False, True]),
                             torch.zeros(2, dtype=torch.bool))
    assert events == ["before", "physics", "after"]


def test_playback_delegates_without_replacing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = make_env()
    closed: list[bool] = []
    lifecycle: list[str] = []
    playback_obs = torch.ones(1, 3)
    original_marker = object()
    playback_marker = object()
    env.marker = original_marker
    temporary = SimpleNamespace(
        num_envs=1,
        marker=playback_marker,
        simulator=SimpleNamespace(
            prepare_playback=lambda _render_mode: lifecycle.append("prepare"),
            terminate_playback=lambda: lifecycle.append("terminate"),
        ),
        render_mode="rgb_array",
        playback_env_index=0,
        render_fps=50.0,
        reset=lambda: playback_obs,
        render=lambda: None,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(env, "_create_playback_environment", lambda: temporary)

    env.prepare_playback()
    assert lifecycle == ["prepare"]
    assert env.num_envs == 1
    assert env.reset() is playback_obs
    assert env.marker is playback_marker
    with pytest.raises(RuntimeError, match="Cannot modify"):
        env.marker = object()
    with pytest.raises(RuntimeError, match="Cannot delete"):
        del env.marker
    assert env.render_mode == "rgb_array"
    assert env.render_fps == 50.0

    env.terminate_playback()
    assert lifecycle == ["prepare", "terminate"]
    assert closed == [True]
    assert env.num_envs == 2
    assert env.marker is original_marker


def test_playback_prepare_failure_clears_temporary_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = make_env()
    lifecycle: list[str] = []

    def fail_prepare(_render_mode: str | None) -> None:
        raise RuntimeError("prepare failed")

    temporary = SimpleNamespace(
        num_envs=1,
        simulator=SimpleNamespace(
            prepare_playback=fail_prepare,
            terminate_playback=lambda: lifecycle.append("terminate"),
        ),
        close=lambda: lifecycle.append("close"),
    )
    monkeypatch.setattr(env, "_create_playback_environment", lambda: temporary)

    with pytest.raises(RuntimeError, match="prepare failed"):
        env.prepare_playback()

    assert lifecycle == ["terminate", "close"]
    assert env._playback_environment is None
    assert env.num_envs == 2


def test_step_uses_old_command_for_reward_and_new_command_for_observation():
    env = make_env()
    task = cast(FakeTask, env.task)
    initial_obs = env.reset()
    torch.testing.assert_close(initial_obs, torch.zeros(2, 3))

    next_obs, transition_obs, reward, terminated, truncated, _ = env.step(
        torch.tensor([[0.2], [0.4]])
    )

    torch.testing.assert_close(reward, torch.zeros(2))
    assert task.last_state_command is not None
    torch.testing.assert_close(task.last_state_command, torch.zeros(2, 1))
    torch.testing.assert_close(transition_obs[:, 0], torch.ones(2))
    torch.testing.assert_close(transition_obs[:, 1], torch.ones(2))
    torch.testing.assert_close(
        transition_obs[:, 2],
        torch.tensor([0.2, 0.4]),
    )
    torch.testing.assert_close(next_obs, transition_obs)
    assert not torch.any(terminated)
    assert not torch.any(truncated)


def test_action_dim_comes_from_action_manager_input() -> None:
    env = make_env()

    assert env.action_dim == 1


def test_render_fps_matches_control_frequency() -> None:
    env = make_env()

    assert env.render_fps == pytest.approx(50.0)


def test_reward_is_scaled_for_consumers_but_not_info() -> None:
    env = make_env()
    task = cast(FakeTask, env.task)
    task.command[:, 0] = torch.tensor([2.0, 3.0])

    _, _, reward, _, _, info = env.step(torch.zeros(2, 1))

    expected_scaled = torch.tensor([0.04, 0.06])
    torch.testing.assert_close(reward, expected_scaled)
    assert task.last_step_result is not None
    torch.testing.assert_close(task.last_step_result.reward, expected_scaled)
    torch.testing.assert_close(info["reward/test"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(info["reward"], torch.tensor(2.5))


def test_step_collects_metric_snapshots_after_done_reset(monkeypatch):
    env = make_env(max_episode_steps=1)
    task = cast(FakeTask, env.task)
    original_reset = task.reset
    revision = torch.zeros(())

    def reset(env_ids=None):
        original_reset(env_ids)
        revision.add_(1)

    def collect_metrics():
        assert torch.count_nonzero(env.current_episode_steps) == 0
        return {"curriculum/test/revision": revision.clone()}

    monkeypatch.setattr(task, "reset", reset)
    monkeypatch.setattr(task, "collect_metrics", collect_metrics)
    env.reset()
    first_info = env.step(torch.zeros(2, 1))[-1]
    second_info = env.step(torch.zeros(2, 1))[-1]
    assert first_info["curriculum/test/revision"].item() == 2.0
    assert second_info["curriculum/test/revision"].item() == 3.0


def test_done_env_returns_terminal_and_reset_observations_separately():
    env = make_env(max_episode_steps=1)
    env.reset()

    next_obs, transition_obs, _, _, truncated, _ = env.step(
        torch.tensor([[0.2], [0.4]])
    )

    assert torch.all(truncated)
    torch.testing.assert_close(
        transition_obs,
        torch.tensor([[1.0, 1.0, 0.2], [1.0, 1.0, 0.4]]),
    )
    torch.testing.assert_close(next_obs, torch.zeros(2, 3))
    assert torch.count_nonzero(env.current_episode_steps) == 0


def test_empty_requirements_support_legacy_backend_and_release_only_old_resources(runtime_context):
    env = VectorEnv(runtime_context)
    env.num_envs = 2
    env.task = BaseTaskLogic(runtime_context)
    env.task.prepare({})
    calls = []

    def legacy_config_update(component, num_envs):
        calls.append(num_envs)

    env.simulator = cast(BaseSimulator, SimpleNamespace(config_update=legacy_config_update))
    component = Component(None, None, None, None, None, None)
    env._build_simulator(component)
    assert calls == [2]
    env.task = cast(BaseTaskLogic, SimpleNamespace(get_simulator_requirements=lambda: {"custom_resource": 7}))
    configurations = []
    env.simulator = cast(BaseSimulator, SimpleNamespace(config_update=lambda **kwargs: configurations.append(kwargs)))
    env._build_simulator(component)
    env.task = BaseTaskLogic(runtime_context)
    env.task.prepare({})
    env._build_simulator(component)
    env._build_simulator(component)
    assert configurations[-2] == {"component": component, "num_envs": 2, "custom_resource": None}
    assert configurations[-1] == {"component": component, "num_envs": 2}


def test_new_backend_does_not_receive_previous_backend_resource_clears(runtime_context, monkeypatch):
    class LegacyBackend:
        def __init__(self, context):
            pass

        def config_update(self, component, num_envs):
            assert num_envs == 2

    env = VectorEnv(runtime_context)
    env.num_envs = 2
    env.task = BaseTaskLogic(runtime_context)
    env.task.prepare({})
    env._simulator_requirement_keys = {"old_resource"}
    monkeypatch.setattr("envs.vector_env.SIM_TYPE_MAP", {"legacy": LegacyBackend})
    monkeypatch.setattr("envs.vector_env.load_yaml", lambda path: {})
    env._build_simulator(Component(None, None, None, None, ComponentInfo("legacy", Path("legacy.yaml")), None))
    assert isinstance(env.simulator, LegacyBackend)
    assert env._simulator_requirement_keys == set()
