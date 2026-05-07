#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_param_sensitivity_hotpotqa8k.sh [max_samples] [min_context_len] [top_k_by_length]

MAX_SAMPLES=${1:-30}
MIN_CONTEXT_LENGTH=${2:-8000}
TOP_K_BY_LENGTH=${3:-100}

OUT_DIR="results"
LOG_DIR="logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

MODEL_SNAPSHOT_DEFAULT="/home/jdk/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-${MODEL_SNAPSHOT_DEFAULT}}"

run_one() {
  local tag="$1"
  local pressure_offset="$2"
  local pressure_scale="$3"
  local evict_base="$4"
  local evict_span="$5"
  local comp_base="$6"
  local comp_span="$7"
  local sink_keep="$8"

  local ts
  ts=$(date +%Y%m%dT%H%M%S)
  local log_file="${LOG_DIR}/param_${tag}_${ts}.log"

  echo "[INFO] Running param set=${tag} max_samples=${MAX_SAMPLES} min_len=${MIN_CONTEXT_LENGTH} top_k=${TOP_K_BY_LENGTH}"

  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  AHEC_PRESSURE_OFFSET_TOKENS="${pressure_offset}" \
  AHEC_PRESSURE_SCALE_TOKENS="${pressure_scale}" \
  AHEC_EVICT_BASE="${evict_base}" \
  AHEC_EVICT_SPAN="${evict_span}" \
  AHEC_COMPRESS_BASE="${comp_base}" \
  AHEC_COMPRESS_SPAN="${comp_span}" \
  AHEC_SINK_KEEP="${sink_keep}" \
  python3 src/evaluate.py \
    --mode generate \
    --config configs/benchmark.param_ahec_only.yaml \
    --output-dir "${OUT_DIR}" \
    --dataset-source longbench \
    --longbench-subsets "hotpotqa" \
    --longbench-min-context-length "${MIN_CONTEXT_LENGTH}" \
    --longbench-sort-by-length-desc \
    --longbench-top-k-by-length "${TOP_K_BY_LENGTH}" \
    --max-samples "${MAX_SAMPLES}" \
    --local-model-name "${LOCAL_MODEL_NAME}" \
    > "${log_file}" 2>&1

  local gen_file
  gen_file=$(grep -oE 'results/generation_eval_[0-9TZ]+\.json' "${log_file}" | tail -n1 || true)
  if [[ -z "${gen_file}" ]]; then
    echo "[ERROR] ${tag} generation output file not found in log: ${log_file}" | tee -a "${log_file}"
    return 1
  fi

  local judge_try=1
  local judge_max_try=3
  while (( judge_try <= judge_max_try )); do
    echo "[INFO] ${tag} judge try=${judge_try}/${judge_max_try} input=${gen_file}" | tee -a "${log_file}"
    if python3 src/evaluate.py \
      --mode judge \
      --config configs/benchmark.param_ahec_only.yaml \
      --input-json "${gen_file}" \
      --output-dir "${OUT_DIR}" \
      --judge-model "MiniMax-Text-01" \
      --judge-repeats 1 \
      >> "${log_file}" 2>&1; then
      break
    fi
    ((judge_try++))
    if (( judge_try > judge_max_try )); then
      echo "[ERROR] ${tag} judge failed after retries" | tee -a "${log_file}"
      return 1
    fi
    sleep 20
  done

  echo "[OK] ${tag} done. log=${log_file}"
}

# low: conservative policy, retain more tokens
run_one "low" 2000 9000 0.08 0.22 0.05 0.28 2

# mid: current default-ish
run_one "mid" 1500 6000 0.15 0.35 0.10 0.45 1

# high: aggressive policy, stronger reduction
run_one "high" 1000 4000 0.22 0.50 0.18 0.55 0

echo "[DONE] parameter sensitivity runs completed."
