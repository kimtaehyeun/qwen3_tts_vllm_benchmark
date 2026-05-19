#!/usr/bin/env bash
set -euo pipefail

echo "Installing RunPod benchmark dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sox ffmpeg
fi

python -m pip install --upgrade pip
python -m pip install --upgrade -r "$REPO_ROOT/requirements.txt"

if command -v python >/dev/null 2>&1; then
  python -c 'import sys, platform; print("Python version:", sys.version.replace("\n", " "))'
fi

python - <<'PY'
import importlib, subprocess, sys
packages = [
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('vllm', 'vllm'),
    ('vllm_omni', 'vllm_omni'),
]
for name, module in packages:
    try:
        m = importlib.import_module(module)
        print(f"{name} version: {getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name} import failed: {exc}")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
except Exception:
    pass

try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
    print('nvidia-smi output:')
    print(result.stdout)
except Exception as exc:
    print('nvidia-smi unavailable or failed:', exc)
PY
