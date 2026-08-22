import torch
import numpy as np
from typing import Any

from app.utils.context import RuntimeContext
from envs.base import BaseEnv
from envs.simulators.base import BaseSimulator
from envs.simulators.utils.context import ModelContext
from envs.simulators.registry import SIM_TYPE_MAP
from envs.tasks.base import BaseTaskLogic
from envs.tasks.utils.context import TaskStepResult
from envs.tasks.registry import TASK_TYPE_MAP
from utils.component import Component
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
        self.max_episode_steps: int
        self.current_episode_steps: torch.Tensor


    def config_update(
        self,
        component: Component,
        num_envs: int,
        max_episode_steps: int,
    ) -> None:

        update_attributes(
            self,
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
        )
        self.current_episode_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.context.device,
        )
        self._build_simulator(component=component)
        model_context = self.simulator.model_context
        self._build_task(
            component=component,
            model_context=model_context,
        )


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
        model_context: ModelContext,
    ) -> None:
        
        if component.task is None:
            if self.task is None:
                raise RuntimeError(
                    f"task instance is required."
                )
            self.task.config_update(
                component=component,
                num_envs=self.num_envs,
                model_context=model_context,
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
                model_context=model_context,
                **task_config
            )


    def reset(self) -> torch.Tensor:

        self.simulator.reset()
        self.task.reset()
        self.current_episode_steps.zero_()
        state = self.simulator.get_state()
        task_context = self.task.build_task_context(
            state=state,
            episode_step=self.current_episode_steps,
        )
        obs, _ = self.task.compute_observation(task_context)

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
        self.current_episode_steps[env_ids] = 0

        state = self.simulator.get_state(env_ids)
        task_context = self.task.build_task_context(
            state=state,
            episode_step=self.current_episode_steps[env_ids],
            env_ids=env_ids,
        )
        reset_obs, _ = self.task.compute_observation(
            task_context,
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

        pre_step_info = self.task.pre_step()

        control, action_info = self.task.process_action(action)
        self.simulator.step(control)
        self.current_episode_steps += 1

        state = self.simulator.get_state()
        task_context = self.task.build_task_context(
            state=state,
            episode_step=self.current_episode_steps,
        ) # old_command, action, last_action

        reward, reward_info = self.task.compute_reward(task_context)
        terminated, terminated_info = self.task.check_terminated(task_context)
        truncated = (self.current_episode_steps >= self.max_episode_steps)

        step_result = TaskStepResult(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        post_step_info = self.task.post_step(task_context, step_result)

        next_task_context = self.task.build_task_context(
            state=state,
            episode_step=self.current_episode_steps,
        ) # new_command, action, last_action

        transition_next_obs, obs_info = self.task.compute_observation(next_task_context)

        next_obs = self._reset_done_envs(
            transition_next_obs,
            terminated,
            truncated,
        )

        info = (
            pre_step_info
            | action_info
            | post_step_info
            | obs_info
            | reward_info
            | terminated_info
            | state
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
