from rl.algorithms.ppo import PPO


ALG_TYPE_MAP: dict[str, type[PPO]] = {
    "ppo": PPO,
}