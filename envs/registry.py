from envs.base import BaseEnv
from envs.vector_env import VectEnv


ENV_TYPE_MAP: dict[str, BaseEnv] = {
    "vector_env": VectEnv,
}
