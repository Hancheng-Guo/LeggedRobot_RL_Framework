import torch


def resolve_device(device_name: str) -> torch.device:
    """check the device with the spcified name"""

    requested_device = torch.device(device_name)

    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device_name!r}, but PyTorch cannot use CUDA."
            )

    return requested_device


# def move_to_device(
#     value: torch.Tensor,
#     device: torch.device,
#     dtype: torch.dtype | None = None,
# ) -> torch.Tensor:
#     """move tensor to specified device, and transfer float type (optional)."""

#     if dtype is None:
#         return value.to(device=device)

#     return value.to(device=device, dtype=dtype)