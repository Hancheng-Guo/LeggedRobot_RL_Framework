from rl.policies.base import BasePolicy
from rl.policies.configurable_actor_critic import ConfigurableActorCritic


POLICY_TYPE_MAP: dict[str, type[BasePolicy]] = {
    "actor_critic": ConfigurableActorCritic,
}
