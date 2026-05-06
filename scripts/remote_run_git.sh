#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-guoguang}
REMOTE_DIR=${REMOTE_DIR:-~/projects/long-context-compression-eviction}
BRANCH=${1:-main}
CONFIG=${2:-configs/benchmark.longbench.yaml}
SAMPLES_FILE=${3:-data/longbench_multi_document_qa.sample.jsonl}
MAX_SAMPLES=${4:-1}

ssh "${REMOTE_HOST}" "
  set -euo pipefail
  cd ${REMOTE_DIR}
  git fetch origin
  git checkout ${BRANCH}
  git pull --ff-only origin ${BRANCH}
  echo '[INFO] commit=' \$(git rev-parse --short HEAD)
  bash scripts/run_eval_real.sh ${CONFIG} ${SAMPLES_FILE} ${MAX_SAMPLES}
"
