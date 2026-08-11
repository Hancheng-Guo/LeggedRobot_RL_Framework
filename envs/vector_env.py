from envs.base import BaseEnv
from utils.component import Component
from app.utils.context import RuntimeContext
from utils.param import update_attributes


class VectorEnv(BaseEnv):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context
        self.simulator = None
        self.task = None

        self.sim_dt = None
        self.frame_skip = None


    def config_update(
        self,
        component: Component,
        sim_dt: float | None = None,
        frame_skip: int | None = None,
    ) -> None:
        
        update_attributes(
            self,
            sim_dt=sim_dt,
            frame_skip=frame_skip,
        )
        self._build_simulator(component=component)
        self._build_task(component=component)


    def _build_simulator(
        self,
        component: Component, 
    ) -> None:
        
        self.simulator = None


    def _build_task(
        self,
        component: Component, 
    ) -> None:
        
        self.task = None


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