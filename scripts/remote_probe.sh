#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-guoguang}
ssh "${REMOTE_HOST}" '
  echo "[HOST] $(hostname)"
  echo "[USER] $(whoami)"
  echo "[GPU]" && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "[PY]" && (python3 --version || true)
  echo "[CONDA]" && (conda --version || true)
'
