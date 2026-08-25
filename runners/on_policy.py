import warnings
import torch
import numpy as np

from app.utils.context import RuntimeContext
from runners.base import BaseRunner
from runners.callbacks.base import BaseCallback
from runners.callbacks.stage import StageCallback
from runners.callbacks.registry import CALLBACK_TYPE_MAP
from runners.utils.frames import save_frames_to_video
from envs.base import BaseEnv
from envs.registry import ENV_TYPE_MAP
from rl.algorithms.base import OnPolicyAlgorithm
from rl.algorithms.registry import ALG_TYPE_MAP
from utils.component import Component
from utils.config import load_yaml
from utils.param import update_attributes


class OnPolicyRunner(BaseRunner):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context
        self.environment: BaseEnv
        self.algorithm: OnPolicyAlgorithm
        
        self.current_iteration: int

        self.max_iterations: int
        self.rollout_length: int
        self.callbacks: list[BaseCallback] = []
        self.stop_callback: list[BaseCallback] = []


    def config_update(
        self,
        component: Component, 
        max_iterations: int | None = None,
        rollout_length: int | None = None,
        callback_names: list[str] | None = None,
    ) -> None:

        self._merge_component(component=component)
        update_attributes(
            self,
            max_iterations=max_iterations,
            rollout_length=rollout_length,
        )
        self._build_callbacks(callback_names=callback_names)
        self._build_environment(component=component)
        self._build_algorithm(component=component)


    def _merge_component(
        self,
        component: Component
    ) -> None:
        
        previous = getattr(self, "_component", None)
        if previous is None:
            self._component = component
            return

        self._component = Component(
            runner=component.runner or previous.runner,
            algorithm=component.algorithm or previous.algorithm,
            policy=component.policy or previous.policy,
            environment=component.environment or previous.environment,
            simulator=component.simulator or previous.simulator,
            task=component.task or previous.task,
        )


    def _create_temporary_environment(
        self,
        num_envs: int,
    ) -> BaseEnv:
        
        if num_envs <= 0:
            raise ValueError("'num_envs' must be greater than 0.")
        if not hasattr(self, "_component"):
            raise RuntimeError("Runner component configuration is missing.")

        environment_info = self._component.environment
        if environment_info is None:
            raise RuntimeError("Environment configuration is missing.")
        if environment_info.type not in ENV_TYPE_MAP:
            raise ValueError(
                f"Invalid environment type: {environment_info.type!r}."
            )

        environment_config = load_yaml(environment_info.config)
        environment_config["num_envs"] = num_envs
        environment = ENV_TYPE_MAP[environment_info.type](
            context=self.context,
        )
        environment.config_update(
            component=self._component,
            **environment_config,
        )
        return environment


    def stage_update(
        self,
        stage_callback: StageCallback | None,
    ) -> None:
        
        self.callbacks = [
            callback
            for callback in self.callbacks
            if not isinstance(callback, StageCallback)
        ]

        if stage_callback is not None:
            self.callbacks.append(stage_callback)


    def _build_callbacks(
        self,
        callback_names: list[str] | None,
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
            if (
                hasattr(self, "environment") is False
                or self.environment is None
            ):
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
            if (
                hasattr(self, "environment") is False
                or not isinstance(self.environment, env_type)
            ):
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
            if (
                hasattr(self, "algorithm") is False
                or self.algorithm is None
            ):
                raise RuntimeError(
                    f"algorithm instance is required."
                )
            self.algorithm.config_update(
                component=component,
                action_dim=self.environment.action_dim,
            )

        else:
            alg_type_name = component.algorithm.type
            if alg_type_name not in ALG_TYPE_MAP:
                raise ValueError(
                    f"Invalid algorithm type: {alg_type_name!r}. "
                )
            
            alg_type = ALG_TYPE_MAP[alg_type_name]
            if not issubclass(alg_type, OnPolicyAlgorithm):
                raise TypeError(
                    f"Algorithm {alg_type_name!r} is not "
                    "an on-policy algorithm."
                )
            alg_config = load_yaml(component.algorithm.config)
            if (
                not hasattr(self, "algorithm")
                or not isinstance(self.algorithm, alg_type)
            ):
                self.algorithm = alg_type(
                    context=self.context,
                )
            self.algorithm.config_update(
                component=component,
                action_dim=self.environment.action_dim,
                **alg_config
            )


    def _run_callbacks(self, hook: str, *args, **kwargs) -> bool:

        should_continue = True

        for callback in self.callbacks:
            method = getattr(callback, hook, None)

            if method is None:
                continue

            if method(*args, **kwargs) is False:
                self.stop_callback.append(callback)
                should_continue = False

        return should_continue
        

    def train(self) -> None:

        if self.environment is None:
            raise RuntimeError("environment is not instantiated.")

        if self.algorithm is None:
            raise RuntimeError("algorithm is not instantiated.")

        if self.max_iterations is None:
            raise RuntimeError("'max_iterations' is missing.")

        if self.rollout_length is None:
            raise RuntimeError("'rollout_length' is missing.")

        self.algorithm.set_train_mode()
        obs = self.environment.reset()
        self.stop_callback.clear()
        
        if not self._run_callbacks("_on_train_start"):
            self._run_callbacks("_on_train_end")
            return

        for iteration in range(self.max_iterations):

            self.current_iteration = iteration
            if not self._run_callbacks("_on_iteration_start"):
                break

            callback_step_break = False
            collected_steps = 0
            for _ in range(self.rollout_length):

                if not self._run_callbacks("_on_step_start"):
                    callback_step_break = True
                    break
                
                with torch.no_grad():
                    policy_output = self.algorithm.act(
                        obs=obs,
                        deterministic=False,
                    )

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
                collected_steps += 1

                if not self._run_callbacks("_on_step_end", info=info):
                    callback_step_break = True
                    break

            if collected_steps == 0:
                break

            # Bootstrap the final observation and compute
            # returns / advantages inside the algorithm.
            with torch.no_grad():
                self.algorithm.compute_returns(last_obs=obs)

            update_info = self.algorithm.update()

            if not self._run_callbacks("_on_iteration_end", info=update_info):
                break

            if callback_step_break:
                break

        self._run_callbacks("_on_train_end")


    def test(
        self,
        num_episodes: int = 1000,
    ) -> None:

        if (
            not isinstance(num_episodes, int)
            or isinstance(num_episodes, bool)
            or num_episodes <= 0
        ):
            raise ValueError("'num_episodes' must be a positive integer.")

        if self.environment is None:
            raise RuntimeError("environment is not instantiated.")

        if self.algorithm is None:
            raise RuntimeError("algorithm is not instantiated.")

        self.algorithm.set_eval_mode()
        obs = self.environment.reset()

        if not self._run_callbacks("_on_test_start"):
            self._run_callbacks("_on_test_end")
            return

        episode_rewards = torch.zeros(
            self.environment.num_envs,
            device=self.context.device,
        )

        episode_lengths = torch.zeros(
            self.environment.num_envs,
            device=self.context.device,
        )

        completed_rewards: list[float] = []
        completed_lengths: list[int] = []

        while len(completed_rewards) < num_episodes:

            if not self._run_callbacks("_on_step_start"):
                break

            with torch.no_grad():
                policy_output = self.algorithm.act(
                    obs,
                    deterministic=True,
                )

            (
                next_obs,
                _,
                reward,
                terminated,
                truncated,
                info,
            ) = self.environment.step(policy_output.action)

            episode_rewards += reward
            episode_lengths += 1

            done = terminated | truncated

            done_ids = done.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                self.algorithm.reset_policy_state(done_ids)

            for env_id in done_ids:

                completed_rewards.append(
                    episode_rewards[env_id].item()
                )

                completed_lengths.append(
                    int(episode_lengths[env_id].item())
                )

                episode_rewards[env_id] = 0.0
                episode_lengths[env_id] = 0

                if len(completed_rewards) >= num_episodes:
                    break

            obs = next_obs

            if not self._run_callbacks("_on_step_end", info=info):
                break

        test_info = {
            "mean_reward": sum(completed_rewards) / len(completed_rewards),
            "mean_episode_length": (
                sum(completed_lengths) / len(completed_lengths)
            ),
            "episode_rewards": completed_rewards,
            "episode_lengths": completed_lengths,
            "num_episodes": len(completed_rewards),
        }

        self._run_callbacks("_on_test_end", info=test_info)


    def play(
        self,
        num_steps: int = 5000
    ) -> None:
        if self.environment is None:
            raise RuntimeError("environment is not instantiated.")

        if self.algorithm is None:
            raise RuntimeError("algorithm is not instantiated.")

        if (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps <= 0
        ):
            raise ValueError("'num_steps' must be a positive integer.")

        original_environment = self.environment
        temporary_environment = self._create_temporary_environment(
            num_envs=1,
        )
        self.environment = temporary_environment

        try:
            self._play_steps(num_steps)
        finally:
            self.algorithm.reset_policy_state()
            temporary_environment.close()
            self.environment = original_environment


    def _play_steps(
        self,
        num_steps: int
    ) -> None:
        
        self.algorithm.set_eval_mode()
        obs = self.environment.reset()

        if not self._run_callbacks("_on_play_start"):
            self._run_callbacks("_on_play_end")
            return

        frames: list[np.ndarray] = []
        step = 0
        while step < num_steps:

            if not self._run_callbacks("_on_step_start"):
                break

            with torch.no_grad():
                policy_output = self.algorithm.act(
                    obs,
                    deterministic=True,
                )

            (
                next_obs,
                _,
                _,
                terminated,
                truncated,
                info,
            ) = self.environment.step(policy_output.action)

            done_ids = (terminated | truncated).nonzero(
                as_tuple=False,
            ).flatten()
            if done_ids.numel() > 0:
                self.algorithm.reset_policy_state(done_ids)

            frame = self.environment.render()
            if frame is not None:
                frames.append(frame)

            obs = next_obs

            if not self._run_callbacks("_on_step_end", info=info):
                break
            step += 1

        if len(frames):
            save_frames_to_video(frames)

        self._run_callbacks("_on_play_end")


    def close(self) -> None:

        self._run_callbacks("_on_close")

        if hasattr(self, "environment"):
            self.environment.close()
            del self.environment

        if hasattr(self, "algorithm"):
            self.algorithm.close()
            del self.algorithm


    def save(self) -> None:
        raise NotImplementedError
