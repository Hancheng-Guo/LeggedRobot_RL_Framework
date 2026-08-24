import torch.nn as nn
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from rl.policies.modules.recurrent import StatefulGRU


MODULE_TYPE_MAP: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "layer_norm": nn.LayerNorm,
    "dropout": nn.Dropout,
    "identity": nn.Identity,
    "gru": StatefulGRU,
}


ModuleType = TypeVar("ModuleType", bound=nn.Module)


def register_module(
    name: str,
) -> Callable[[type[ModuleType]], type[ModuleType]]:
    
    if not name:
        raise ValueError("Module registration name cannot be empty.")

    def decorator(
        module_type: type[ModuleType],
    ) -> type[ModuleType]:
        if name in MODULE_TYPE_MAP:
            raise ValueError(f"Module type {name!r} is already registered.")
        MODULE_TYPE_MAP[name] = module_type
        return module_type

    return decorator


def build_module(
    config: Mapping[str, Any],
    variables: Mapping[str, Any] | None = None,
) -> nn.Module:
    
    module_config = dict(config)
    module_type_name = module_config.pop("type", None)

    if not isinstance(module_type_name, str) or not module_type_name:
        raise ValueError("Module config requires a non-empty 'type'.")
    if module_type_name not in MODULE_TYPE_MAP:
        valid_names = ", ".join(MODULE_TYPE_MAP)
        raise ValueError(
            f"Invalid module type: {module_type_name!r}. "
            f"Expected one of: {valid_names}."
        )

    nested_params = module_config.pop("params", {})
    if not isinstance(nested_params, Mapping):
        raise TypeError("Module 'params' must be a mapping.")
    duplicate_names = set(module_config).intersection(nested_params)
    if duplicate_names:
        names = ", ".join(sorted(duplicate_names))
        raise ValueError(f"Duplicate module parameters: {names}.")
    module_config.update(nested_params)

    if variables is not None:
        module_config = {
            name: variables.get(value, value)
            if isinstance(value, str)
            else value
            for name, value in module_config.items()
        }

    module_type = MODULE_TYPE_MAP[module_type_name]
    try:
        return module_type(**module_config)
    except TypeError as error:
        raise TypeError(
            f"Invalid parameters for module {module_type_name!r}: "
            f"{module_config!r}."
        ) from error
