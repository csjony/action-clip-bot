#!/usr/bin/env bash
# =============================================================================
# Cron entrypoint — runs one full pipeline job.
# Intended to be invoked by cron (see scripts/setup.sh output).
#
#   scripts/run_job.sh             # normal run (generates full video + publishes)
#   scripts/run_job.sh --quick-run # fast smoke-test: 3 clips × 5s = 15s, no publish
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

mkdir -p data

# Lock so overlapping cron runs (if a job runs long) don't clobber each other.
LOCKFILE="data/run.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "[$(date -u +%FT%TZ)] Another run is in progress — skipping." >&2
    exit 0
fi

echo "[$(date -u +%FT%TZ)] action-clip-bot job starting"
"$PY" -m src.pipeline "$@"
echo "[$(date -u +%FT%TZ)] action-clip-bot job finished"
