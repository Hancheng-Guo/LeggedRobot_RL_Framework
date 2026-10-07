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
            prepare_playback=lambda: lifecycle.append("prepare"),
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

    def fail_prepare() -> None:
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
