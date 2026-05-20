#!/usr/bin/env bash
set -euo pipefail
PID_FILE="logs/server.pid"

stop_pid() {
  local pid="$1"
  if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping server pid=$pid"
    kill "$pid" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        return 0
      fi
      sleep 0.5
    done
    echo "Force stopping server pid=$pid"
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
}

stop_pattern() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Stopping leftover processes matching: $pattern"
    pkill -TERM -f "$pattern" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "$pattern" 2>/dev/null || true
  fi
}

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  stop_pid "$PID"
  rm -f "$PID_FILE"
else
  echo "PID file not found: $PID_FILE"
fi

stop_pattern "vllm-omni serve"
stop_pattern "VLLM::Worker"
stop_pattern "VLLM::EngineCore"

exit 0
