#!/usr/bin/env bash
# =============================================================================
# VPS bootstrap for action-clip-bot.
# Tested on Ubuntu 22.04 / 24.04 (Hetzner CX22). Run once as a non-root user
# with sudo. Idempotent — safe to re-run.
#
#   bash scripts/setup.sh
#
# Sets up: system deps (ffmpeg, python, fonts), Python venv, project install,
# and prints the cron line to add for the daily job.
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> [1/5] System packages (ffmpeg, python, fonts, sqlite)"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    ffmpeg \
    python3 python3-venv python3-pip \
    fonts-dejavu-core fonts-liberation \
    sqlite3 \
    curl ca-certificates

echo "==> [2/5] Python venv at .venv/"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# Use the venv's python directly. We avoid `source activate` because its
# embedded absolute paths break if the venv is later moved or renamed.
PY=".venv/bin/python"

"$PY" -m pip install --upgrade pip wheel >/dev/null
"$PY" -m pip install -e ".[dev,dashboard]"

echo "==> [3/5] Project directories"
mkdir -p data config

echo "==> [4/5] Secrets check"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "    Created .env from .env.example — EDIT IT and add your API keys."
else
    echo "    .env already exists — leaving as-is."
fi

echo "==> [5/5] Smoke test"
"$PY" -c "import src; from src.config import get_settings; print('config OK ->', get_settings().data_dir)"

cat <<EOF

------------------------------------------------------------
✅ Setup complete.

Next steps:
  1. Edit .env and fill in API keys (Gemini, MiniMax, PixVerse, etc.)
  2. Test one full pipeline run manually:
       python -m src.pipeline --dry-run
  3. Add the cron entry below (crontab -e) for unattended runs
     (Mon/Wed/Fri at 09:00 server time):
       0 9 * * 1,3,5  $PROJECT_DIR/scripts/run_job.sh >> $PROJECT_DIR/data/cron.log 2>&1
  4. Watch Telegram for the post links + monthly spend.
------------------------------------------------------------
EOF
