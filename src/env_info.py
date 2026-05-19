import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional


def _import_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
        return getattr(module, '__version__', None) or str(module)
    except Exception:
        return None


def _run_nvidia_smi() -> Optional[str]:
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def gather_environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': sys.version.replace('\n', ' '),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'nvidia_smi': _run_nvidia_smi(),
        'torch_version': _import_version('torch'),
        'transformers_version': _import_version('transformers'),
        'vllm_version': _import_version('vllm'),
        'vllm_omni_version': _import_version('vllm_omni'),
        'gpu_count': None,
        'gpu_names': [],
        'nvidia_driver_version': None,
        'env': {k: v for k, v in os.environ.items() if k.startswith('CUDA') or k.startswith('NVIDIA')},
    }

    try:
        import torch

        info['cuda_available'] = torch.cuda.is_available()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        info['cuda_available'] = False

    if info['nvidia_smi']:
        for line in info['nvidia_smi'].splitlines():
            if 'Driver Version' in line:
                info['nvidia_driver_version'] = line.split('Driver Version')[-1].strip().strip(':').strip()
                break

    return info
