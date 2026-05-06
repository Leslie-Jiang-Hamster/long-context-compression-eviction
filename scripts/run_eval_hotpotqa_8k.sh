#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/benchmark.longbench.yaml}
MAX_SAMPLES=${2:-143}
MIN_CONTEXT_LENGTH=${3:-8000}

echo "[INFO] Running HOTPOTQA-only semantic eval (len >= ${MIN_CONTEXT_LENGTH}) with config: ${CONFIG}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p results
python3 src/evaluate.py \
  --config "${CONFIG}" \
  --output-dir results \
  --dataset-source longbench \
  --longbench-subsets "hotpotqa" \
  --longbench-min-context-length "${MIN_CONTEXT_LENGTH}" \
  --max-samples "${MAX_SAMPLES}" \
  --local-model-name "Qwen/Qwen2.5-7B-Instruct" \
  --judge-model "MiniMax-Text-01"
