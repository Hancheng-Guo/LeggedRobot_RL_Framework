from rl.algorithms.base import BaseAlgorithm
from rl.algorithms.ppo import PPO


ALG_TYPE_MAP: dict[str, BaseAlgorithm] = {
    "ppo": PPO,
}