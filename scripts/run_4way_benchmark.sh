#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fourway_${STAMP}}"
LIMIT="${LIMIT:-300}"
WARMUP="${WARMUP:-5}"
TORCH_BATCH_SIZE="${TORCH_BATCH_SIZE:-1}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
SAVE_AUDIO_FLAG="${SAVE_AUDIO_FLAG:---save-audio}"

mkdir -p "$OUTPUT_ROOT/logs"

stop_server_if_running() {
  if [ -f logs/server.pid ]; then
    bash scripts/stop_server.sh || true
  fi
}

wait_for_vllm() {
  local api_base="$1"
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 3 "${api_base}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "vLLM server did not become ready: ${api_base}" >&2
  return 1
}

run_torch() {
  local quant="$1"
  local out_dir="$2"
  echo "=== torch ${quant} -> ${out_dir} ==="
  python -u -m src.benchmark_torch \
    --config configs/qwen3_tts_base.yaml \
    --output-dir "$out_dir" \
    --quantization "$quant" \
    --warmup-requests "$WARMUP" \
    --batch-size "$TORCH_BATCH_SIZE" \
    --limit "$LIMIT" \
    $SAVE_AUDIO_FLAG \
    2>&1 | tee "${OUTPUT_ROOT}/logs/$(basename "$out_dir").log"
}

run_vllm() {
  local config_path="$1"
  local out_dir="$2"
  local api_base
  api_base="$(python - "$config_path" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('server', {}).get('api_base', 'http://127.0.0.1:8091'))
PY
)"
  echo "=== vLLM ${config_path} -> ${out_dir} ==="
  stop_server_if_running
  bash scripts/launch_server.sh "$config_path" 2>&1 | tee "${OUTPUT_ROOT}/logs/$(basename "$out_dir")_server_launch.log"
  wait_for_vllm "$api_base"
  python -u -m src.benchmark_manifest \
    --config "$config_path" \
    --output-dir "$out_dir" \
    --warmup-requests "$WARMUP" \
    --concurrency "$CONCURRENCY" \
    --batch-size "$VLLM_BATCH_SIZE" \
    --limit "$LIMIT" \
    --timeout-sec "$TIMEOUT_SEC" \
    $SAVE_AUDIO_FLAG \
    --response-format wav \
    2>&1 | tee "${OUTPUT_ROOT}/logs/$(basename "$out_dir").log"
  stop_server_if_running
}

echo "Output root: ${OUTPUT_ROOT}"
echo "LIMIT=${LIMIT} WARMUP=${WARMUP} TORCH_BATCH_SIZE=${TORCH_BATCH_SIZE} VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE} CONCURRENCY=${CONCURRENCY}"

stop_server_if_running

run_torch none "${OUTPUT_ROOT}/01_torch_bf16"
run_vllm configs/qwen3_tts_base.yaml "${OUTPUT_ROOT}/02_vllm_bf16"
run_torch bnb4 "${OUTPUT_ROOT}/03_torch_llm4_mtp4"
run_vllm configs/qwen3_tts_bnb4.yaml "${OUTPUT_ROOT}/04_vllm_llm4_mtp4"

python -m src.compare_benchmarks \
  --run-dir "${OUTPUT_ROOT}/01_torch_bf16" \
  --run-dir "${OUTPUT_ROOT}/02_vllm_bf16" \
  --run-dir "${OUTPUT_ROOT}/03_torch_llm4_mtp4" \
  --run-dir "${OUTPUT_ROOT}/04_vllm_llm4_mtp4" \
  --output-dir "${OUTPUT_ROOT}/comparison"

echo "4-way benchmark complete: ${OUTPUT_ROOT}"
