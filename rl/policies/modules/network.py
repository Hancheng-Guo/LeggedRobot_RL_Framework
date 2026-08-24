from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rl.policies.modules.recurrent import StatefulGRU
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


    @property
    def is_recurrent(self) -> bool:
        return any(isinstance(module, StatefulGRU) for module in self)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._forward_step(input)


    def forward_sequence(
        self,
        inputs: torch.Tensor,
        initial_state: dict[str, torch.Tensor],
        reset_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        if inputs.ndim < 3:
            raise ValueError(
                "Sequence input must have shape [time, batch, ...]."
            )
        if reset_mask.shape != inputs.shape[:2]:
            raise ValueError("reset_mask must have shape [time, batch].")
        if reset_mask.dtype != torch.bool:
            raise TypeError("reset_mask must have dtype torch.bool.")
        if inputs.shape[0] == 0:
            raise ValueError("Sequence input cannot be empty.")

        self._validate_recurrent_state(initial_state)
        state = dict(initial_state)
        outputs: list[torch.Tensor] = []

        for step in range(inputs.shape[0]):
            keep_mask = (~reset_mask[step]).unsqueeze(-1)
            state = {
                name: value * keep_mask.to(dtype=value.dtype)
                for name, value in state.items()
            }
            output, state = self._forward_with_state(
                inputs=inputs[step],
                state=state,
            )
            outputs.append(output)

        return torch.stack(outputs), state


    def get_recurrent_state(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        
        return {
            name: module.get_state(batch_size, device, dtype)
            for name, module in self.named_children()
            if isinstance(module, StatefulGRU)
        }


    def reset_recurrent_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        
        for module in self:
            if isinstance(module, StatefulGRU):
                module.reset_state(env_ids)


    def _forward_step(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        
        value = inputs
        for module in self:
            if isinstance(module, StatefulGRU):
                value = module.forward_step(value)
            else:
                value = module(value)
        return value


    def _forward_with_state(
        self,
        inputs: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        value = inputs
        next_state = dict(state)
        for name, module in self.named_children():
            if isinstance(module, StatefulGRU):
                value, next_state[name] = module.forward_with_state(
                    inputs=value,
                    state=next_state[name],
                )
            else:
                value = module(value)
        return value, next_state


    def _validate_recurrent_state(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        
        expected_names = {
            name
            for name, module in self.named_children()
            if isinstance(module, StatefulGRU)
        }
        if set(state) != expected_names:
            raise ValueError("Recurrent state keys do not match the network.")
