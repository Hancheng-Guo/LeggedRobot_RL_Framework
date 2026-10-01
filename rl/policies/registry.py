from .base import BasePolicy
from .configurable_actor_critic import ConfigurableActorCritic


POLICY_TYPE_MAP: dict[str, type[BasePolicy]] = {
    "actor_critic": ConfigurableActorCritic,
}
