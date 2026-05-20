#!/usr/bin/env bash
# run_local_mac_manifest.sh
# Run full manifest inference on local Mac with Qwen3-TTS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m src.benchmark_local_mac \
  --config configs/qwen3_tts_local_mac.yaml \
  --output-dir ~/Workspace/results/qwen3_tts_vllm_benchmark/local_mac_full \
  --batch-size 1 \
  --warmup-requests 1 \
  --save-audio

echo ""
echo "Full manifest inference complete."
echo "Results: ~/Workspace/results/qwen3_tts_vllm_benchmark/local_mac_full"
