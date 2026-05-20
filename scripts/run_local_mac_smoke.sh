#!/usr/bin/env bash
# run_local_mac_smoke.sh
# Run a single-sample smoke test of local Mac Qwen3-TTS inference.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m src.benchmark_local_mac \
  --config configs/qwen3_tts_local_mac.yaml \
  --output-dir ~/Workspace/results/qwen3_tts_vllm_benchmark/local_mac_smoke \
  --limit 1 \
  --batch-size 1 \
  --warmup-requests 0 \
  --save-audio

echo ""
echo "Smoke test complete."
echo "Results: ~/Workspace/results/qwen3_tts_vllm_benchmark/local_mac_smoke"
