import warnings
import torch
from pathlib import Path
from typing import Any

from runners.base import BaseRunner
from runners.callbacks.stage import StageCallback
from runners.callbacks.registry import CALLBACK_TYPE_MAP
from utils.component import Component
from app.utils.context import RuntimeContext
from envs.registry import ENV_TYPE_MAP
from rl.algorithms.registry import ALG_TYPE_MAP
from utils.config import load_yaml, get_yaml_value
from utils.param import update_attributes


class OnPolicyRunner(BaseRunner):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context
        self.environment = None
        self.algorithm = None
        
        self.current_iteration = None

        self.max_iterations = None
        self.rollout_length = None
        self.callbacks = []


    def config_update(
        self,
        component: Component, 
        max_iterations: int | None = None,
        rollout_length: int | None = None,
        callback_names: list[str] | None = None,
    ) -> None:

        update_attributes(
            self,
            max_iterations=max_iterations,
            rollout_length=rollout_length,
        )
        self._build_callbacks(callback_names=callback_names)
        self._build_environment(component=component)
        self._build_algorithm(component=component)


    def _build_callbacks(
        self,
        callback_names: list[str],
    ) -> None:

        if callback_names is not None:
            self.callbacks = []

            for callback_name in callback_names:
                if callback_name not in CALLBACK_TYPE_MAP:
                    warnings.warn(
                        f"Unknown callback type: '{callback_name}'. "
                    )
                else:
                    self.callbacks.append(
                        CALLBACK_TYPE_MAP[callback_name](
                            runner=self,
                            max_iterations=self.max_iterations,
                            rollout_length=self.rollout_length,
                            context=self.context,
                        )
                    )


    def _build_environment(
        self,
        component: Component
    ) -> None:
        
        if component.environment is None:
            if self.environment is None:
                raise RuntimeError(
                    f"environment instance is required."
                )
            self.environment.config_update(component=component)

        else:
            env_type_name = component.environment.type
            if env_type_name not in ENV_TYPE_MAP:
                raise ValueError(
                    f"Invalid environment type: {env_type_name!r}. "
                )
            
            env_type = ENV_TYPE_MAP[env_type_name]
            env_config = load_yaml(component.environment.config)
            if not isinstance(self.environment, env_type):
                self.environment = env_type(
                    context=self.context
                )
            self.environment.config_update(
                component=component,
                **env_config
            )


    def _build_algorithm(
        self,
        component: Component
    ) -> None:
        
        if component.algorithm is None:
            if self.algorithm is None:
                raise RuntimeError(
                    f"algorithm instance is required."
                )
            self.algorithm.config_update(component=component)

        else:
            alg_type_name = component.algorithm.type
            if alg_type_name not in ALG_TYPE_MAP:
                raise ValueError(
                    f"Invalid algorithm type: {alg_type_name!r}. "
                )
            
            alg_type = ALG_TYPE_MAP[alg_type_name]
            alg_config = load_yaml(component.algorithm.config)
            if not isinstance(self.algorithm, alg_type):
                self.algorithm = alg_type(
                    context=self.context,
                )
            self.algorithm.config_update(
                component=component,
                **alg_config
            )


    def _run_callbacks(self, hook: str, *args, **kwargs) -> None:

        for callback in self.callbacks:
            method = getattr(callback, hook, None)

            if method is None:
                continue

            method(*args, **kwargs)
        

    def train(self) -> None:

        obs = self.environment.reset()

        self._run_callbacks("_on_train_start")

        for iteration in range(self.max_iterations):

            self.current_iteration = iteration
            self._run_callbacks("_on_iteration_start")

            for _ in range(self.rollout_length):

                self._run_callbacks("_on_step_start")
                
                with torch.no_grad():
                    policy_output = self.algorithm.act(obs)

                (
                    next_obs,
                    transition_next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.environment.step(policy_output.action)

                self.algorithm.process_transition(
                    obs=obs,
                    policy_output=policy_output,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    next_obs=transition_next_obs,
                    info=info,
                )

                obs = next_obs

                self._run_callbacks("_on_step_end")

            # Bootstrap the final observation and compute
            # returns / advantages inside the algorithm.
            with torch.no_grad():
                self.algorithm.compute_returns(last_obs=obs)

            update_info = self.algorithm.update()

            self._run_callbacks("_on_iteration_end", info=update_info)

        self._run_callbacks("_on_train_end")


    def test(self) -> None:
        # obs = self.env.reset()

        # while True:
        #     action = self.algorithm.act(obs, deterministic=True)
        #     obs, reward, done, info = self.env.step(action)

        #     if done:
        #         break
        pass


    def play(self) -> None:
        pass


    def close(self) -> None:
        pass


    def stage_update(
        self,
        stage_callback: StageCallback
    ):

        for callback in self.callbacks:
            if isinstance(callback, StageCallback):
                self.callbacks.remove(callback)

        self.callbacks.append(stage_callback)