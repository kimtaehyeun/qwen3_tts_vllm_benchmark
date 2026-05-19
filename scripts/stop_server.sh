#!/usr/bin/env bash
set -euo pipefail
PID_FILE="logs/server.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "PID file not found: $PID_FILE"
  exit 1
fi
PID=$(cat "$PID_FILE")
if [ -z "$PID" ]; then
  echo "PID file is empty"
  exit 1
fi
if kill -0 "$PID" >/dev/null 2>&1; then
  echo "Stopping server pid=$PID"
  kill "$PID"
  rm -f "$PID_FILE"
  exit 0
else
  echo "Process $PID not running"
  rm -f "$PID_FILE"
  exit 1
fi
