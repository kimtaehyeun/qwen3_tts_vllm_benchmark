import gc
import platform
import sys
from typing import Any, Dict

import torch


def get_best_device(requested_device: str = "auto") -> torch.device:
    """Return the best available device: cuda > mps > cpu."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def get_recommended_dtype(device: torch.device, requested_dtype: str = "auto") -> torch.dtype:
    """Return a stable dtype for the given device.

    cuda  -> bfloat16 (or config override)
    mps   -> float16  (float32 fallback handled at load time)
    cpu   -> float32
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if requested_dtype != "auto" and requested_dtype in dtype_map:
        return dtype_map[requested_dtype]

    device_type = device.type
    if device_type == "cuda":
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def describe_local_torch_environment() -> Dict[str, Any]:
    """Return a dict describing the local torch/hardware environment."""
    mps_built = False
    mps_available = False
    try:
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()
    except Exception:
        pass

    cuda_available = False
    cuda_device_count = 0
    cuda_device_names = []
    try:
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
        cuda_device_names = [torch.cuda.get_device_name(i) for i in range(cuda_device_count)]
    except Exception:
        pass

    return {
        "torch_version": torch.__version__,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
    }


def clear_device_cache(device: torch.device) -> None:
    """Free cached memory for the given device."""
    device_type = device.type
    if device_type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    elif device_type == "mps":
        gc.collect()
        try:
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
    else:
        gc.collect()
