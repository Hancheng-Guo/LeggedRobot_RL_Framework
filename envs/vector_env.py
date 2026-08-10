from envs.base import BaseEnv


class VectEnv(BaseEnv):

    def __init__(
        self,
        simulator_detail: dict,
        task_detail: dict,
        configs: list[dict],
    ) -> None:
        self.configs = configs


    def reset(self):

        obs = self.simulator.reset()
        return self.observation_manager.process(obs)


    def step(
        self,
        action,
    ):

        self.simulator.step(action)
        obs = self.observation_manager.get()
        reward = self.reward_manager.compute()
        terminated = self.termination_manager.check()
        truncated = self.timeout_manager.check()

        info = {
            "reward": reward,
        }

        transition_next_obs = obs.clone()

        obs = self._reset_done_envs(
            obs,
            terminated,
            truncated,
        )

        return (
            obs,
            transition_next_obs,
            reward,
            terminated,
            truncated,
            info,
        )