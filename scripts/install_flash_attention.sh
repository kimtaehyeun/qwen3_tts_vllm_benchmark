#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/qwen3_tts_base_optimized.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

read_config() {
  python - "$CONFIG_PATH" "$1" <<'PY'
import sys
import yaml

config_path, key = sys.argv[1], sys.argv[2]
with open(config_path, encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
node = cfg
for part in key.split('.'):
    if not isinstance(node, dict):
        node = None
        break
    node = node.get(part)
if isinstance(node, bool):
    print(str(node).lower())
elif node is None:
    print('')
elif isinstance(node, list):
    print(' '.join(str(item) for item in node))
else:
    print(str(node))
PY
}

INSTALL="$(read_config torch_runtime.flash_attention.install)"
if [ "${INSTALL:-true}" = "false" ]; then
  echo "FlashAttention install disabled by config: ${CONFIG_PATH}"
  exit 0
fi

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('flash_attn') else 1)
PY
then
  python - <<'PY'
import flash_attn
print(f"flash_attn already installed: {getattr(flash_attn, '__version__', 'unknown')}")
PY
  exit 0
fi

PACKAGE="$(read_config torch_runtime.flash_attention.package)"
VERSION="$(read_config torch_runtime.flash_attention.version)"
NO_BUILD_ISOLATION="$(read_config torch_runtime.flash_attention.no_build_isolation)"
NO_CACHE_DIR="$(read_config torch_runtime.flash_attention.no_cache_dir)"
MAX_JOBS_CFG="$(read_config torch_runtime.flash_attention.max_jobs)"
CUDA_ARCH_LIST_CFG="$(read_config torch_runtime.flash_attention.cuda_arch_list)"
EXTRA_PIP_ARGS="$(read_config torch_runtime.flash_attention.extra_pip_args)"
SHOW_PROGRESS="$(read_config torch_runtime.flash_attention.show_progress)"
PROGRESS_INTERVAL="$(read_config torch_runtime.flash_attention.progress_interval_sec)"
LOG_DIR_CFG="$(read_config torch_runtime.flash_attention.log_dir)"

PACKAGE="${PACKAGE:-flash-attn}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
LOG_DIR_CFG="${LOG_DIR_CFG:-logs}"
mkdir -p "$LOG_DIR_CFG"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR_CFG}/flash_attention_install_${STAMP}.log"

if [ -n "$VERSION" ]; then
  INSTALL_TARGET="${PACKAGE}==${VERSION}"
else
  INSTALL_TARGET="${PACKAGE}"
fi

PIP_ARGS=(python -m pip install "$INSTALL_TARGET")
if [ "${NO_BUILD_ISOLATION:-true}" != "false" ]; then
  PIP_ARGS+=(--no-build-isolation)
fi
if [ "${NO_CACHE_DIR:-true}" != "false" ]; then
  PIP_ARGS+=(--no-cache-dir)
fi
if [ -n "$EXTRA_PIP_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=($EXTRA_PIP_ARGS)
  PIP_ARGS+=("${EXTRA_ARRAY[@]}")
fi

export MAX_JOBS="${MAX_JOBS:-${MAX_JOBS_CFG:-4}}"
if [ -n "$CUDA_ARCH_LIST_CFG" ]; then
  export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST_CFG"
fi

echo "Installing FlashAttention for torch runtime"
echo "Config     : $CONFIG_PATH"
echo "Target     : $INSTALL_TARGET"
echo "MAX_JOBS   : ${MAX_JOBS}"
echo "ARCH_LIST  : ${TORCH_CUDA_ARCH_LIST:-auto}"
echo "Log        : $LOG_PATH"
echo "Command    : ${PIP_ARGS[*]}"

if [ "${SHOW_PROGRESS:-true}" = "false" ]; then
  "${PIP_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
else
  (
    while true; do
      sleep "$PROGRESS_INTERVAL"
      echo ""
      echo "[flash-attn progress] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd \
        | grep -E 'pip install flash|flash-attn|ninja -v|nvcc .*flash_attn|cicc .*flash' \
        | grep -v grep \
        | head -20 || true
      echo "[flash-attn progress] disk:"
      df -h / /tmp | tail -n +1 || true
      echo ""
    done
  ) &
  PROGRESS_PID=$!
  set +e
  "${PIP_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
  STATUS=${PIPESTATUS[0]}
  set -e
  kill "$PROGRESS_PID" >/dev/null 2>&1 || true
  wait "$PROGRESS_PID" 2>/dev/null || true
  if [ "$STATUS" -ne 0 ]; then
    echo "FlashAttention install failed. See log: $LOG_PATH" >&2
    exit "$STATUS"
  fi
fi

python - <<'PY'
import flash_attn
print(f"flash_attn installed: {getattr(flash_attn, '__version__', 'unknown')}")
PY
