#!/usr/bin/env bash
# setup_mac_local.sh
# Install minimum dependencies for local Mac Qwen3-TTS inference.
# Does NOT install vllm, vllm-omni, bitsandbytes, flash-attn, or flashinfer.
set -euo pipefail

# ── Check that a virtual environment is active ────────────────────────────
if [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: No active conda environment or virtualenv detected."
    echo "         It is strongly recommended to activate one first, e.g.:"
    echo "           conda activate qwen3_tts_mac"
    echo "         or"
    echo "           source .venv/bin/activate"
    echo ""
    read -r -p "Continue without an active environment? [y/N] " reply
    case "$reply" in
        [Yy]*) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

echo "=== [1/5] Upgrading pip ==="
pip install --upgrade pip

echo ""
echo "=== [2/5] Installing PyTorch (Apple Silicon MPS) ==="
pip install torch torchvision torchaudio

echo ""
echo "=== [3/5] Installing HuggingFace stack ==="
pip install transformers accelerate huggingface_hub

echo ""
echo "=== [4/5] Installing qwen-tts ==="
pip install qwen-tts

echo ""
echo "=== [5/5] Installing audio / data / utility packages ==="
pip install numpy pandas soundfile librosa scipy pyyaml tqdm psutil rich

echo ""
echo "=== Checking MPS availability ==="
python - <<'PY'
import sys

def check_mps():
    try:
        import torch
    except ImportError:
        print("ERROR: torch import failed after installation.")
        sys.exit(1)

    print(f"  torch version : {torch.__version__}")
    print(f"  MPS built     : {torch.backends.mps.is_built()}")
    print(f"  MPS available : {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        print("  -> Apple Silicon GPU acceleration is ENABLED (mps)")
    elif torch.backends.mps.is_built():
        print("  -> MPS is built but NOT available.")
        print("     Requires macOS 12.3 (Monterey) or later and Apple Silicon.")
        print("     Will fall back to CPU.")
    else:
        print("  -> MPS is not built. Will use CPU.")

if __name__ == "__main__":
    check_mps()
PY

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Place reference_manifest.csv in ~/Workspace/datasets/qwen3_tts_vllm_benchmark/"
echo "  2. Place reference audio files in ~/Workspace/datasets/qwen3_tts_vllm_benchmark/"
echo "  3. Run smoke test:"
echo "       bash scripts/run_local_mac_smoke.sh"
