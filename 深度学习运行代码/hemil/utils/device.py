from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible")
    return torch.device(spec)


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return f"{props.name} ({props.total_memory / 1024**3:.1f} GB)"
    return "cpu"
