#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=========================================="
echo "[1/2] MLX bf16 base (300 samples)"
echo "=========================================="
conda run --no-capture-output -n mlx-llm python3 -m src.benchmark_mlx \
  --config configs/qwen3_tts_mlx.yaml \
  --output-dir ~/Workspace/results/qwen3_tts_vllm_benchmark/mlx_base_full \
  --warmup-requests 1 \
  --save-audio

echo ""
echo "=========================================="
echo "[2/2] MLX int4 4-bit (300 samples)"
echo "=========================================="
conda run --no-capture-output -n mlx-llm python3 -m src.benchmark_mlx \
  --config configs/qwen3_tts_mlx_4bit.yaml \
  --output-dir ~/Workspace/results/qwen3_tts_vllm_benchmark/mlx_4bit_full \
  --warmup-requests 1 \
  --save-audio

echo ""
echo "=========================================="
echo "All done."
echo "Results:"
echo "  bf16 : ~/Workspace/results/qwen3_tts_vllm_benchmark/mlx_base_full/"
echo "  int4 : ~/Workspace/results/qwen3_tts_vllm_benchmark/mlx_4bit_full/"
echo "=========================================="
