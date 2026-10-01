from .base import BaseAlgorithm
from .ppo import PPO


ALG_TYPE_MAP: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
}
