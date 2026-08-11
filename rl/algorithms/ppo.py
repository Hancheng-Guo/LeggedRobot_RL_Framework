import torch

from utils.component import Component
from app.utils.context import RuntimeContext
from rl.algorithms.base import OnPolicyAlgorithm, PolicyOutput
from utils.param import update_attributes


class PPO(OnPolicyAlgorithm):

    def __init__(
        self,
        context: RuntimeContext,
    ) -> None:

        self.context = context

        self.model = None
        self.storage = None

        self.gamma = None
        self.gae_lambda = None
        self.clip_range = None
        self.entropy_coef = None
        self.value_coef = None
        self.max_grad_norm = None
        self.num_epochs = None
        self.num_mini_batches = None


    def config_update(
        self,
        component: Component,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        entropy_coef: float,
        value_coef: float,
        max_grad_norm: float,
        num_epochs: int,
        num_mini_batches: int,
    ) -> None:

        update_attributes(
            self,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            num_epochs=num_epochs,
            num_mini_batches=num_mini_batches,
        )
        self._build_model(component=component)
        self._build_storage(component=component)
        

    def _build_model(
        self,
        component: Component
    ) -> None:
        pass


    def _build_storage(
        self,
        component: Component
    ) -> None:
        pass
        

    def act(
        self,
        obs,
    ) -> PolicyOutput:

        output = self.model(obs)
        action = output.action
        log_prob = output.log_prob
        value = output.value

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            value=value,
        )


    def eval(self) -> None:
        pass


    def update(self) -> None:
        pass


    def process_transition(
        self,
        obs,
        policy_output,
        reward,
        terminated,
        truncated,
        next_obs,
    ) -> None:

        # self.storage.add(
        #     obs=obs,
        #     action=policy_output.action,
        #     log_prob=policy_output.log_prob,
        #     value=policy_output.value,
        #     reward=reward,
        #     terminated=terminated,
        #     truncated=truncated,
        #     next_obs=next_obs,
        # )
        pass


    def compute_returns(
        self,
        last_obs,
    ) -> None:

        # with torch.no_grad():
        #     value = self.model.value(last_obs)

        # self.storage.compute_returns(last_value=value)
        pass


    def update(self) -> dict:

        # info = {}

        # for epoch in range(self.num_epochs):

        #     for batch in self.storage:

        #         loss = self.compute_loss(batch)
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(
        #             self.model.parameters(),
        #             self.max_grad_norm,
        #         )

        #         self.optimizer.step()

        # self.storage.clear()

        # return info
        pass
