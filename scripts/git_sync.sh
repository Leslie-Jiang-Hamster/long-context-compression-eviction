#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-guoguang}
REMOTE_DIR=${REMOTE_DIR:-~/projects/long-context-compression-eviction}
BRANCH=${1:-main}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[ERROR] Local git has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

echo "[INFO] Pushing local ${BRANCH}..."
git push origin "${BRANCH}"

echo "[INFO] Pulling on remote ${REMOTE_HOST}:${REMOTE_DIR}..."
ssh "${REMOTE_HOST}" "
  set -euo pipefail
  cd ${REMOTE_DIR}
  git fetch origin
  git checkout ${BRANCH}
  git pull --ff-only origin ${BRANCH}
  echo '[OK] Remote synced to' \$(git rev-parse --short HEAD)
"
