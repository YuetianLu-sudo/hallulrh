#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_exp1_gemini_judge_4way.sh /path/to/your/run_dir
#
# The run dir should contain the 4 "*.for_judge.csv" files somewhere within depth<=3.

RUN_DIR="${1:?Please provide RUN_DIR (e.g., /mounts/work/.../runs/experiments/exp1_...)}"

# Tune this if you hit rate-limit (429). With 4 parallel jobs, 0.3~0.6 is usually safer than 0.2.
export GEMINI_SLEEP="${GEMINI_SLEEP:-0.4}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"

OUT_DIR="${RUN_DIR}/judge"
LOG_DIR="${RUN_DIR}/logs_judge"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

MODELS=("gemma_7b_it" "llama3_1_8b_instruct" "mistral_7b_instruct" "qwen2_5_7b_instruct")
GPUS=("0" "1" "2" "3")

find_input_csv () {
  local model_key="$1"
  local run_dir="$2"

  # Support both naming conventions:
  #   gemma_7b_it.for_judge.csv
  #   gemma_7b_it_for_judge.csv
  local a
  a="$(find "$run_dir" -maxdepth 3 -type f -name "${model_key}.for_judge.csv" -print -quit || true)"
  if [[ -n "$a" ]]; then
    echo "$a"
    return 0
  fi

  a="$(find "$run_dir" -maxdepth 3 -type f -name "${model_key}_for_judge.csv" -print -quit || true)"
  if [[ -n "$a" ]]; then
    echo "$a"
    return 0
  fi

  echo ""
  return 0
}

echo "[info] RUN_DIR=${RUN_DIR}"
echo "[info] OUT_DIR=${OUT_DIR}"
echo "[info] LOG_DIR=${LOG_DIR}"
echo "[info] GEMINI_MODEL=${GEMINI_MODEL}"
echo "[info] GEMINI_SLEEP=${GEMINI_SLEEP}"

for i in "${!MODELS[@]}"; do
  m="${MODELS[$i]}"
  gpu="${GPUS[$i]}"

  in_csv="$(find_input_csv "$m" "$RUN_DIR")"
  if [[ -z "$in_csv" ]]; then
    echo "[ERROR] Could not find CSV for model=${m} under ${RUN_DIR}" >&2
    exit 1
  fi

  out_csv="${OUT_DIR}/${m}.with_judge.csv"
  log="${LOG_DIR}/${m}.log"

  echo "[info] Launch: ${m} (gpu=${gpu})"
  echo "       in : ${in_csv}"
  echo "       out: ${out_csv}"
  echo "       log: ${log}"

  # Note: GPU is not used by Gemini judging; CUDA_VISIBLE_DEVICES is only to satisfy your 4-card job allocation.
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -u scripts/lm_judge_gemini.py judge-csv \
    --input "${in_csv}" \
    --output "${out_csv}" \
    --resume \
    > "${log}" 2>&1 &
done

wait
echo "[done] All 4 judge jobs finished."
echo "[done] Outputs are in: ${OUT_DIR}"
