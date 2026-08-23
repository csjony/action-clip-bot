#!/usr/bin/env bash
# =============================================================================
# Dashboard entrypoint — starts the FastAPI server.
#
#   scripts/run_dashboard.sh           # start on port 8080 (or DASHBOARD_PORT)
#   DASHBOARD_PORT=9000 scripts/run_dashboard.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Resolve the Python interpreter: prefer the project venv, fall back to system.
# Calling the venv python directly avoids relying on `activate` (whose embedded
# absolute paths break if the venv directory is ever moved or renamed).
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

# Load .env if present (provides DASHBOARD_USER, DASHBOARD_PASSWORD, etc.)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

mkdir -p data

PORT="${DASHBOARD_PORT:-8080}"
HOST="${DASHBOARD_HOST:-0.0.0.0}"
export DASHBOARD_PORT="$PORT"
export DASHBOARD_HOST="$HOST"

echo "[$(date -u +%FT%TZ)] action-clip-bot dashboard starting on http://${HOST}:${PORT}"
exec "$PY" -m src.dashboard
