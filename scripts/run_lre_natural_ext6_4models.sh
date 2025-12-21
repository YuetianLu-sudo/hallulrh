#!/usr/bin/env bash
set -euo pipefail

PROMPTS="data/lre/natural_relations_ext6.jsonl"
OUT_DIR="data/lre/natural_by_model"
LOG_DIR="logs/lre_natural_ext6"

mkdir -p "$OUT_DIR" "$LOG_DIR"

MODEL_KEYS=("llama3_1_8b_instruct" "gemma_7b_it" "mistral_7b_instruct" "qwen2_5_7b_instruct")
MODEL_IDS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "google/gemma-7b-it" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B-Instruct")
GPUS=(0 1 2 3)

for i in "${!MODEL_KEYS[@]}"; do
  mk="${MODEL_KEYS[$i]}"
  mid="${MODEL_IDS[$i]}"
  gpu="${GPUS[$i]}"
  out="${OUT_DIR}/${mk}.csv"
  log="${LOG_DIR}/${mk}.gpu${gpu}.log"

  echo "[launch] gpu=${gpu} model_key=${mk}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python scripts/run_lre_relpanel.py \
    --model-id "${mid}" \
    --model-name "${mk}" \
    --prompts "${PROMPTS}" \
    --output-csv "${out}" \
    --device cuda \
    >"${log}" 2>&1 &

done

wait
echo "[done] All natural LRE jobs finished."
