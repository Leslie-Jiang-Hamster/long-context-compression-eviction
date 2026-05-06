#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/benchmark.longbench.yaml}
SAMPLES_FILE=${2:-data/longbench_multi_document_qa.sample.jsonl}
MAX_SAMPLES=${3:-2}

echo "[INFO] Running REAL semantic evaluation with config: ${CONFIG}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${SAMPLES_FILE}" ]]; then
  echo "[ERROR] Samples file not found: ${SAMPLES_FILE}" >&2
  exit 1
fi

mkdir -p results
python3 src/evaluate.py \
  --config "${CONFIG}" \
  --output-dir results \
  --samples-file "${SAMPLES_FILE}" \
  --max-samples "${MAX_SAMPLES}" \
  --local-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --judge-model "MiniMax-Text-01"
