import torch
from typing import cast

from envs.simulators.base import BaseSimulator
from envs.tasks.base import BaseTaskLogic
from envs.tasks.utils.context import TaskContext
from envs.vector_env import VectorEnv


class FakeSimulator:

    def __init__(self, num_envs: int) -> None:
        self.state = torch.zeros(num_envs, 1)

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
    ) -> dict[str, torch.Tensor]:
        state = self.state if env_ids is None else self.state[env_ids]
        return {"state": state.clone()}


class FakeTask:

    def __init__(self, num_envs: int) -> None:
        self.command = torch.zeros(num_envs, 1)
        self.action = torch.zeros(num_envs, 1)
        self.last_action = torch.zeros(num_envs, 1)

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
        state: dict[str, torch.Tensor],
        episode_step: torch.Tensor,
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
            action=action,
            last_action=last_action,
            episode_step=episode_step,
        )

    def pre_step(self) -> dict:
        return {}

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
        return task_context.command["target"].squeeze(-1).clone(), {}

    def check_terminated(
        self,
        task_context: TaskContext,
    ) -> tuple[torch.Tensor, dict]:
        return torch.zeros(
            task_context.action.shape[0],
            dtype=torch.bool,
        ), {}

    def post_step(self, task_context: TaskContext) -> dict:
        self.command += 1.0
        return {}

    def compute_observation(
        self,
        task_context: TaskContext,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return torch.cat(
            (
                task_context.state["state"],
                task_context.command["target"],
                task_context.action,
            ),
            dim=-1,
        ), {}


def make_env(max_episode_steps: int = 10) -> VectorEnv:
    env = VectorEnv.__new__(VectorEnv)
    env.num_envs = 2
    env.max_episode_steps = max_episode_steps
    env.current_episode_steps = torch.zeros(2, dtype=torch.long)
    env.simulator = cast(BaseSimulator, FakeSimulator(2))
    env.task = cast(BaseTaskLogic, FakeTask(2))
    return env


def test_step_uses_old_command_for_reward_and_new_command_for_observation():
    env = make_env()
    initial_obs = env.reset()
    torch.testing.assert_close(initial_obs, torch.zeros(2, 3))

    next_obs, transition_obs, reward, terminated, truncated, _ = env.step(
        torch.tensor([[0.2], [0.4]])
    )

    torch.testing.assert_close(reward, torch.zeros(2))
    torch.testing.assert_close(transition_obs[:, 0], torch.ones(2))
    torch.testing.assert_close(transition_obs[:, 1], torch.ones(2))
    torch.testing.assert_close(
        transition_obs[:, 2],
        torch.tensor([0.2, 0.4]),
    )
    torch.testing.assert_close(next_obs, transition_obs)
    assert not torch.any(terminated)
    assert not torch.any(truncated)


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
