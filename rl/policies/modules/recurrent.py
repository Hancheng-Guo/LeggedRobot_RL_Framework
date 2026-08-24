import torch


class StatefulGRU(torch.nn.Module):
    """GRU module with policy-owned state for vectorized environments."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        
        super().__init__()

        if bidirectional:
            raise ValueError(
                "Bidirectional GRU is not valid for online causal policies."
            )
        
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self._hidden_state: torch.Tensor | None
        self.register_buffer(
            "_hidden_state",
            None,
            persistent=False,
        )


    @property
    def state_size(self) -> int:
        return self.num_layers * self.num_directions * self.hidden_size


    def forward(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_step(inputs)


    def forward_step(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        
        if inputs.ndim != 2:
            raise ValueError(
                "StatefulGRU expects one step with shape [batch, features]."
            )

        hidden_state = self._ensure_state(
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
        )
        output, next_state = self.gru(
            inputs.unsqueeze(0),
            hidden_state,
        )
        self._hidden_state = next_state.detach()
        return output.squeeze(0)


    def forward_with_state(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if inputs.ndim != 2:
            raise ValueError(
                "StatefulGRU expects one step with shape [batch, features]."
            )
        if state.shape[0] != inputs.shape[0]:
            raise ValueError("Recurrent state batch size does not match input.")

        hidden_state = self._unflatten_state(state)
        output, next_hidden_state = self.gru(
            inputs.unsqueeze(0),
            hidden_state,
        )
        return output.squeeze(0), self._flatten_state(next_hidden_state)


    def get_state(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        
        if self._hidden_state is None:
            if batch_size is None or device is None or dtype is None:
                raise RuntimeError(
                    "batch_size, device and dtype are required to initialize "
                    "the recurrent state."
                )
            self._ensure_state(batch_size, device, dtype)
        elif (
            batch_size is not None
            and self._hidden_state.shape[1] != batch_size
        ):
            raise ValueError("Recurrent state batch size does not match input.")

        assert self._hidden_state is not None
        return self._flatten_state(self._hidden_state).detach().clone()


    def reset_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        
        if self._hidden_state is None:
            return
        if env_ids is None:
            self._hidden_state = None
            return

        mask = torch.ones(
            self._hidden_state.shape[1],
            device=self._hidden_state.device,
            dtype=self._hidden_state.dtype,
        )
        mask[env_ids] = 0.0
        self._hidden_state = self._hidden_state * mask.view(1, -1, 1)


    def _ensure_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        
        if self._hidden_state is None:
            self._hidden_state = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
        elif self._hidden_state.shape[1] != batch_size:
            raise ValueError("Recurrent state batch size does not match input.")
        return self._hidden_state


    def _flatten_state(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        
        return (
            state.transpose(0, 1)
            .reshape(state.shape[1], self.state_size)
        )


    def _unflatten_state(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        
        if state.ndim != 2 or state.shape[-1] != self.state_size:
            raise ValueError(
                "Recurrent state must have shape "
                f"[batch, {self.state_size}]."
            )
        
        return (
            state.reshape(
                state.shape[0],
                self.num_layers * self.num_directions,
                self.hidden_size,
            )
            .transpose(0, 1)
            .contiguous()
        )
