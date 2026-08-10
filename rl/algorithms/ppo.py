import torch

from rl.algorithms.base import OnPolicyAlgorithm, PolicyOutput


class PPO(OnPolicyAlgorithm):

    def __init__(
        self,
        model_detail: dict,
        configs: list[dict],
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
