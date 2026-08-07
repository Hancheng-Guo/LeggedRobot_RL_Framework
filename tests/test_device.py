import pytest
import torch
from app.utils.device import resolve_device

@pytest.mark.parametrize(
    "device_name, expected_type",
    [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
    ],
)

def test_get_existing_device(device_name, expected_type):
    device = resolve_device(device_name)
    assert device.type == expected_type


def test_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_get_nonexistent_device():
    with pytest.raises(RuntimeError):
        resolve_device("unknown")
