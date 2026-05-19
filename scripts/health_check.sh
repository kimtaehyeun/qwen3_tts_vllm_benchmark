#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <api_base>"
  exit 1
fi
API_BASE="$1"
for endpoint in "/v1/models" "/health" "/metrics"; do
  url="${API_BASE}${endpoint}"
  echo "Checking ${url}"
  python - <<'PY' "$url"
import sys, urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        print(resp.status, resp.reason)
        body = resp.read(500)
        print(body.decode('utf-8', errors='ignore'))
except Exception as exc:
    print('failed:', exc)
    sys.exit(1)
PY
done
