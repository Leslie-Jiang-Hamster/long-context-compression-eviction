#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/benchmark.longbench.yaml}
MAX_SAMPLES=${2:-30}
LONGBENCH_SUBSETS=${3:-hotpotqa,2wikimqa,musique}
MIN_CONTEXT_LENGTH=${4:-4000}

echo "[INFO] Running REAL semantic evaluation with config: ${CONFIG}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi
mkdir -p results
python3 src/evaluate.py \
  --config "${CONFIG}" \
  --output-dir results \
  --dataset-source longbench \
  --longbench-subsets "${LONGBENCH_SUBSETS}" \
  --longbench-min-context-length "${MIN_CONTEXT_LENGTH}" \
  --max-samples "${MAX_SAMPLES}" \
  --local-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --judge-model "MiniMax-Text-01"
