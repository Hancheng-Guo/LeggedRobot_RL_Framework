import warnings
import torch
from pathlib import Path
from typing import Any

from runners.base import BaseRunner
# from app.utils.stage import StageManager
from runners.callbacks.registry import CALLBACK_TYPE_MAP
from utils.component import Component
from envs.registry import ENV_TYPE_MAP
from rl.algorithms.registry import ALG_TYPE_MAP
from utils.config import load_yaml, get_yaml_value


class OnPolicyRunner(BaseRunner):

    def __init__(
        self,
        max_iterations: int,
        rollout_length: int,
        callback_names: list[str],
        component: Component, 
    ) -> None:
        self.max_iterations = max_iterations
        self.current_iteration = None
        self.rollout_length = rollout_length

        self.callbacks = None
        self._build_callbacks(
            callback_names=callback_names,
            max_iterations=max_iterations,
            rollout_length=rollout_length,
        )

        environment_detail = component.get("environment")
        simulator_detail = component.get("simulator")
        task_detail = component.get("task")
        self.environment = None
        self._build_environment(
            environment_detail=environment_detail,
            simulator_detail=simulator_detail,
            task_detail=task_detail,
        )

        algorithm_detail = component.get("algorithm")
        model_detail = component.get("model")
        self.algorithm = None
        self._build_algorithm(
            algorithm_detail=algorithm_detail,
            model_detail=model_detail,
        )


    def _build_callbacks(
        self,
        callback_names: list[str],
        **kwargs
    ) -> None:
        
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
                        **kwargs
                    )
                )


    def _build_environment(
        self,
        environment_detail: dict,
        simulator_detail: dict,
        task_detail: dict,
    ) -> None:

        env_type_name = environment_detail.get("type_name")
        if env_type_name not in ENV_TYPE_MAP:
            raise(
                f"Unknown environment type: '{env_type_name}'. "
            )

        env_config_names = environment_detail.get("config_names")
        env_config_dirs = [
            (
                Path(get_yaml_value("configs/base.yaml", "env_config_dir")) 
                / env_config_name
            )
            for env_config_name in env_config_names
        ]
        env_configs = [
            load_yaml(env_config_dir)
            for env_config_dir in env_config_dirs
        ]

        self.environment = ENV_TYPE_MAP[env_type_name](
            simulator_detail=simulator_detail,
            task_detail=task_detail,
            configs=env_configs,
        )


    def _build_algorithm(
        self,
        algorithm_detail: dict,
        model_detail: dict,
    ) -> None:
        
        alg_type_name = algorithm_detail.get("type_name")
        if alg_type_name not in ALG_TYPE_MAP:
            raise(
                f"Unknown algorithm type: '{alg_type_name}'. "
            )

        alg_config_names = algorithm_detail.get("config_names")
        alg_config_dirs = [
            (
                Path(get_yaml_value("configs/base.yaml", "alg_config_dir")) 
                / alg_config_name
            )
            for alg_config_name in alg_config_names
        ]
        alg_configs = [
            load_yaml(alg_config_dir)
            for alg_config_dir in alg_config_dirs
        ]

        self.algorithm = ALG_TYPE_MAP[alg_type_name](
            model_detail=model_detail,
            configs=alg_configs,
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


    def config_update(
        self,
        max_iterations: int,
        rollout_length: int,
        callback_names: list[str],
        component: dict, 
    ) -> None:
        pass