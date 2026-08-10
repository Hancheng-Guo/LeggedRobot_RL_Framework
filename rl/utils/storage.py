class RolloutStorage:

    def add(
        self,
        obs,
        action,
        log_prob,
        value,
        reward,
        terminated,
        truncated,
        next_obs,
    ):

        self.obs.append(obs)

        self.actions.append(action)

        self.log_probs.append(
            log_prob
        )

        self.values.append(
            value
        )

        self.rewards.append(
            reward
        )

        self.terminated.append(
            terminated
        )

        self.truncated.append(
            truncated
        )


    def compute_returns(
        self,
        last_value,
    ):

        self.returns, self.advantages = (
            compute_gae(
                rewards=self.rewards,
                values=self.values,
                last_value=last_value,
                terminated=self.terminated,
            )
        )


    def clear(self):

        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.advantages.clear()
        self.returns.clear()