#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

conda run -n mlx-llm python3 -m src.benchmark_mlx \
  --config configs/qwen3_tts_mlx.yaml \
  --output-dir ~/Workspace/results/qwen3_tts_vllm_benchmark/mlx_smoke \
  --limit 1 \
  --warmup-requests 0 \
  --save-audio
