#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/benchmark.longbench.yaml}
echo "[INFO] Running evaluation with config: ${CONFIG}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p results
python3 src/evaluate.py \
  --config "${CONFIG}" \
  --output-dir results \
  --samples-file data/longbench_multi_document_qa.sample.jsonl \
  --max-samples 2 \
  --dry-run
