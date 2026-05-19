#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <config.yaml>"
  exit 1
fi

CONFIG_PATH="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

MTP_QUANT=$(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
server = cfg.get('server', {})
print(str(server.get('mtp_quantization', '')).lower())
PY
)
if [ "$MTP_QUANT" = "bitsandbytes" ]; then
  export QWEN3_TTS_BNB4_MTP=1
  export QWEN3_TTS_MTP_QUANT=bitsandbytes
elif [ "$MTP_QUANT" = "int4_weightonly" ] || [ "$MTP_QUANT" = "int4" ]; then
  unset QWEN3_TTS_BNB4_MTP || true
  export QWEN3_TTS_MTP_QUANT=int4_weightonly
else
  unset QWEN3_TTS_BNB4_MTP || true
  unset QWEN3_TTS_MTP_QUANT || true
fi

CMD=$(python - "$CONFIG_PATH" "$@" <<'PY'
import importlib.util
import os
import shlex
import sys

import yaml


def resolve_stage_configs_path(value):
    if not value:
        return None
    value = str(value)
    if os.path.isabs(value) or os.path.exists(value):
        return value
    if value.endswith('.yaml'):
        spec = importlib.util.find_spec('vllm_omni')
        if spec and spec.submodule_search_locations:
            candidate = os.path.join(
                spec.submodule_search_locations[0],
                'model_executor',
                'stage_configs',
                value,
            )
            if os.path.exists(candidate):
                return candidate
    return value


config_path = sys.argv[1]
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
server = cfg.get('server', {})
model_id = server.get('model_id')
if model_id is None:
    raise SystemExit('server.model_id is required')
port = server.get('port', 8091)
host = server.get('host', '0.0.0.0')
gpu_memory_utilization = server.get('gpu_memory_utilization')
tensor_parallel_size = server.get('tensor_parallel_size')
max_num_seqs = server.get('max_num_seqs')
max_model_len = server.get('max_model_len')
trust_remote_code = server.get('trust_remote_code')
stage_configs_path = resolve_stage_configs_path(server.get('stage_configs_path'))
allowed_local_media_path = server.get('allowed_local_media_path')
extra_vllm_args = server.get('extra_vllm_args') or []
args = sys.argv[2:]
executable = server.get('executable', 'vllm')
command = [str(executable), 'serve', model_id]
if server.get('omni'):
    command.append('--omni')
command.extend(['--host', str(host), '--port', str(port)])
if gpu_memory_utilization is not None:
    command.extend(['--gpu-memory-utilization', str(gpu_memory_utilization)])
if tensor_parallel_size is not None:
    command.extend(['--tensor-parallel-size', str(tensor_parallel_size)])
if max_num_seqs is not None:
    command.extend(['--max-num-seqs', str(max_num_seqs)])
if max_model_len is not None:
    command.extend(['--max-model-len', str(max_model_len)])
if stage_configs_path:
    command.extend(['--stage-configs-path', str(stage_configs_path)])
if allowed_local_media_path:
    command.extend(['--allowed-local-media-path', str(allowed_local_media_path)])
if trust_remote_code:
    command.append('--trust-remote-code')
for extra in extra_vllm_args:
    command.append(str(extra))
command.extend(args)
print(' '.join(shlex.quote(str(item)) for item in command))
PY
)

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_DIR/server_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/server.pid"

echo "Launching server with command: $CMD"
if command -v setsid >/dev/null 2>&1; then
  setsid bash -lc "$CMD" >"$LOG_PATH" 2>&1 < /dev/null &
else
  nohup bash -lc "$CMD" >"$LOG_PATH" 2>&1 < /dev/null &
fi
PID=$!
echo "$PID" >"$PID_FILE"
sleep 2
if ! kill -0 "$PID" >/dev/null 2>&1; then
  echo "Server process exited during startup. Last log lines:"
  tail -n 80 "$LOG_PATH" || true
  exit 1
fi
echo "Server started with PID $PID"
echo "Logs: $LOG_PATH"
