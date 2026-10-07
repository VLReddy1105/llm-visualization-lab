from dataclasses import dataclass

import torch

from llm_lab.errors import ServiceError


@dataclass(frozen=True)
class DeviceSelection:
    device: str
    dtype: torch.dtype
    available: tuple[str, ...]
    reason: str


def select_device(requested: str = "auto") -> DeviceSelection:
    available = []
    if torch.cuda.is_available():
        available.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    if torch.backends.mps.is_available():
        available.append("mps")
    available.append("cpu")
    selected = available[0] if requested == "auto" else requested
    if selected not in available:
        raise ServiceError(
            "DEVICE_UNAVAILABLE",
            f"Device '{selected}' is unavailable.",
            503,
            details={"available_devices": available},
        )
    native_bf16 = False
    if selected.startswith("cuda:"):
        with torch.cuda.device(selected):
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if native_bf16:
        dtype = torch.bfloat16
        reason = "CUDA reports native bfloat16 support; output softmax accumulates in float64."
    else:
        dtype = torch.float32
        reason = "Conservative float32 baseline for CPU/MPS or CUDA without native bfloat16."
    return DeviceSelection(selected, dtype, tuple(available), reason)


def synchronize(device: str) -> None:
    if device.startswith("cuda:"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()
