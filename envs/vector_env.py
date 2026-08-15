import torch
import numpy as np
from typing import Any

from envs.base import BaseEnv
from envs.simulators.base import BaseSimulator
from envs.simulators.registry import SIM_TYPE_MAP
from envs.tasks.base import BaseTaskLogic
from envs.tasks.registry import TASK_TYPE_MAP
from utils.component import Component
from app.utils.context import RuntimeContext
from utils.config import load_yaml
from utils.param import update_attributes


class VectorEnv(BaseEnv):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context
        self.simulator: BaseSimulator
        self.task: BaseTaskLogic

        self.num_envs: int


    def config_update(
        self,
        component: Component,
        num_envs: int = 1,
    ) -> None:

        update_attributes(
            self,
            num_envs=num_envs,
        )
        self._build_simulator(component=component)
        self._build_task(component=component)


    def _build_simulator(
        self,
        component: Component
    ) -> None:
        
        if component.simulator is None:
            if self.simulator is None:
                raise RuntimeError(
                    f"simulator instance is required."
                )
            self.simulator.config_update(
                component=component,
                num_envs=self.num_envs,
            )

        else:
            sim_type_name = component.simulator.type
            if sim_type_name not in SIM_TYPE_MAP:
                raise ValueError(
                    f"Invalid simulator type: {sim_type_name!r}. "
                )
            
            sim_type = SIM_TYPE_MAP[sim_type_name]
            sim_config = load_yaml(component.simulator.config)
            if (
                not hasattr(self, "simulator")
                or not isinstance(self.simulator, sim_type)
            ):
                self.simulator = sim_type(context=self.context)
            self.simulator.config_update(
                component=component,
                num_envs=self.num_envs,
                **sim_config
            )


    def _build_task(
        self,
        component: Component, 
    ) -> None:
        
        if component.task is None:
            if self.task is None:
                raise RuntimeError(
                    f"task instance is required."
                )
            self.task.config_update(
                component=component,
                num_envs=self.num_envs,
            )

        else:
            task_type_name = component.task.type
            if task_type_name not in TASK_TYPE_MAP:
                raise ValueError(
                    f"Invalid task type: {task_type_name!r}. "
                )
            
            task_type = TASK_TYPE_MAP[task_type_name]
            task_config = load_yaml(component.task.config)
            if (
                not hasattr(self, "task")
                or not isinstance(self.task, task_type)
            ):
                self.task = task_type(context=self.context)
            self.task.config_update(
                component=component,
                num_envs=self.num_envs,
                **task_config
            )


    def reset(self) -> torch.Tensor:

        self.simulator.reset()
        self.task.reset()
        state = self.simulator.get_state()
        obs = self.task.compute_observation(state)

        return obs


    def _reset_done_envs(
        self,
        obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:

        done = terminated | truncated

        if not torch.any(done):
            return obs

        env_ids = torch.nonzero(
            done,
            as_tuple=False,
        ).squeeze(-1)

        self.simulator.reset(env_ids)
        self.task.reset(env_ids)

        state = self.simulator.get_state(env_ids)
        reset_obs = self.task.compute_observation(
            state,
            env_ids=env_ids,
        )

        next_obs = obs.clone()
        next_obs[env_ids] = reset_obs

        return next_obs


    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:

        self.simulator.step(action)
        state = self.simulator.get_state()
        obs = self.task.compute_observation(state)
        reward = self.task.compute_reward(state)
        terminated = self.task.check_terminated(state)
        truncated = self.task.check_truncated(state)

        info = {
            "reward": reward,
        }

        transition_next_obs = obs

        next_obs = self._reset_done_envs(
            obs,
            terminated,
            truncated,
        )

        return (
            next_obs,
            transition_next_obs,
            reward,
            terminated,
            truncated,
            info,
        )


    def close(self) -> None:
        
        if self.simulator is not None:
            self.simulator.close()


    def render(self) -> np.ndarray | None:
        return self.simulator.render()