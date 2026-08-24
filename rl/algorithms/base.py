from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from torch import Tensor

from app.utils.context import RuntimeContext
from rl.policies.base import BasePolicy, RecurrentPolicy
from rl.utils.storage import RolloutStorage


@dataclass(slots=True)
class PolicyOutput:

    action: Tensor
    log_prob: Tensor
    value: Tensor


class BaseAlgorithm(ABC):

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        self.context = context
        self.policy: BasePolicy
        self.storage: RolloutStorage


    @abstractmethod
    def config_update(
        self,
        *args, **kwargs
    ) -> None:
        raise NotImplementedError


    @abstractmethod
    def act(
        self,
        obs: Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        raise NotImplementedError


    @abstractmethod
    def update(self) -> dict[str, float]:
        """Perform one optimization phase and return scalar diagnostics."""
        raise NotImplementedError


    @abstractmethod
    def close(self) -> None:
        """Release resources owned by the algorithm."""
        raise NotImplementedError
    

    def set_train_mode(self) -> None:

        if not hasattr(self, "policy"):
            raise RuntimeError(
                "setting train mode should init policy first"
            )
        
        self.policy.set_train_mode()
        if (
            isinstance(self.policy, RecurrentPolicy)
            and self.policy.is_recurrent
        ):
            self.policy.reset_recurrent_state()


    def set_eval_mode(self) -> None:

        if not hasattr(self, "policy"):
            raise RuntimeError(
                "setting train mode should init policy first"
            )
        
        self.policy.set_eval_mode()
        if (
            isinstance(self.policy, RecurrentPolicy)
            and self.policy.is_recurrent
        ):
            self.policy.reset_recurrent_state()


    def reset_policy_state(self, env_ids: Tensor | None = None) -> None:
        """Reset episode-local policy state for selected environments."""
        if (
            hasattr(self, "policy")
            and isinstance(self.policy, RecurrentPolicy)
            and self.policy.is_recurrent
        ):
            self.policy.reset_recurrent_state(env_ids)


class OnPolicyAlgorithm(BaseAlgorithm):
    """Contract for algorithms trained from a freshly collected rollout."""

    @abstractmethod
    def process_transition(
        self,
        obs: Tensor,
        policy_output: PolicyOutput,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        next_obs: Tensor,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Store one vectorized transition.

        ``next_obs`` is the transition observation before auto-reset. This is
        required to bootstrap time-limit truncations correctly.
        """
        raise NotImplementedError


    @abstractmethod
    def compute_returns(
        self,
        last_obs: Tensor
    ) -> None:
        """Finalize returns and advantages for the current rollout."""
        raise NotImplementedError


class OffPolicyAlgorithm(BaseAlgorithm):
    """Contract for algorithms trained from replayed transitions."""

    @abstractmethod
    def sample_batch(self) -> Any:
        raise NotImplementedError
