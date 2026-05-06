#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-guoguang}
REMOTE_DIR=${REMOTE_DIR:-~/projects/long-context-compression-eviction}

rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'results' \
  ./ "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "[OK] Synced to ${REMOTE_HOST}:${REMOTE_DIR}"
