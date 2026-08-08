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
        return super().reset()


    def step(self, actions):
        return super().step(actions)