#!/usr/bin/env bash
# 启动翻译网关（FastAPI）。
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
WORKERS="${WORKERS:-1}"

echo "启动网关: host=${HOST} port=${PORT} workers=${WORKERS}"
exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --workers "${WORKERS}"
