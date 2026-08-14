from envs.vector_env import VectorEnv


ENV_TYPE_MAP: dict[str, type[VectorEnv]] = {
    "vector_env": VectorEnv,
}
