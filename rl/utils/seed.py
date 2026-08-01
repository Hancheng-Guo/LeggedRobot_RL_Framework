import os
import random
import torch
import secrets
import numpy as np


def set_seed(
    seed: int | str | None = None,
    deterministic: bool = False,
) -> None:

    if seed is None:
        seed = secrets.randbits(32)
    if isinstance(seed, str):
        try:
            seed = int(seed)
        except ValueError:
            raise ValueError("seed string cannot tranfer to integer.")
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)