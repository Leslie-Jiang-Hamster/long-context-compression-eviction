#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-guoguang}
REMOTE_DIR=${REMOTE_DIR:-~/projects/long-context-compression-eviction}
CONFIG=${1:-configs/benchmark.longbench.yaml}

ssh "${REMOTE_HOST}" "
  set -euo pipefail
  cd ${REMOTE_DIR}
  echo '[INFO] pwd='\$(pwd)
  bash scripts/run_eval.sh ${CONFIG}
"
