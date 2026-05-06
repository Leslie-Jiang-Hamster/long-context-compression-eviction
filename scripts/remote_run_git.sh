#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-guoguang}
REMOTE_DIR=${REMOTE_DIR:-~/projects/long-context-compression-eviction}
BRANCH=${1:-main}
CONFIG=${2:-configs/benchmark.longbench.yaml}
MAX_SAMPLES=${3:-30}
LONGBENCH_SUBSETS=${4:-hotpotqa,2wikimqa,musique}
MIN_CONTEXT_LENGTH=${5:-4000}

ssh "${REMOTE_HOST}" "
  set -euo pipefail
  cd ${REMOTE_DIR}
  git fetch origin
  git checkout ${BRANCH}
  git pull --ff-only origin ${BRANCH}
  echo '[INFO] commit=' \$(git rev-parse --short HEAD)
  bash scripts/run_eval_real.sh ${CONFIG} ${MAX_SAMPLES} ${LONGBENCH_SUBSETS} ${MIN_CONTEXT_LENGTH}
"
