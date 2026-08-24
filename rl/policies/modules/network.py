from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rl.policies.modules.registry import build_module


class ConfigurableNetwork(torch.nn.Sequential):

    def __init__(
        self,
        modules: Sequence[Mapping[str, Any]],
        variables: Mapping[str, Any] | None = None,
    ) -> None:
        
        if not modules:
            raise ValueError("Network 'modules' must be a non-empty sequence.")

        super().__init__(*(
            build_module(config, variables=variables)
            for config in modules
        ))
