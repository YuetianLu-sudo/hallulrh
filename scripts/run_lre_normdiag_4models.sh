#!/usr/bin/env bash
set -euo pipefail

PROMPTS="data/lre/natural_relations_ext6_q_fname.jsonl"
OUTDIR="analysis/lre_normdiag_20251224"

MODEL_KEYS=("llama3_1_8b_instruct" "gemma_7b_it" "mistral_7b_instruct" "qwen2_5_7b_instruct")
MODEL_IDS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "google/gemma-7b-it" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B-Instruct")

for gpu in 0 1 2 3; do
  (
    key="${MODEL_KEYS[$gpu]}"
    mid="${MODEL_IDS[$gpu]}"
    out="$OUTDIR/${key}.csv"
    echo "[gpu $gpu] $key -> $out"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_lre_relpanel_normdiag.py \
      --model-id "$mid" \
      --model-name "$key" \
      --prompts "$PROMPTS" \
      --output-csv "$out" \
      --device "cuda:0"
    echo "[gpu $gpu] done: $key"
  ) &
done
wait
echo "[done] normdiag finished: $OUTDIR"
