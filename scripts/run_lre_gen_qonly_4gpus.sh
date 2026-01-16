#!/usr/bin/env bash
set -euo pipefail

PROMPTS="data/lre_hernandez/prompts/lre_prompts_qonly.jsonl"
RUN_TAG="${RUN_TAG:-lre_qonly_$(date +%Y%m%d_%H%M%S)}"
OUTDIR="data/lre_hernandez/gen/${RUN_TAG}"
LOGDIR="logs/lre_gen_${RUN_TAG}"

SYS_PROMPT="${SYS_PROMPT:-You are a helpful assistant. Answer with a single short phrase.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

mkdir -p "$OUTDIR" "$LOGDIR"

count_lines() { wc -l "$1" | awk '{print $1}'; }
N=$(count_lines "$PROMPTS")

is_complete_jsonl() {
  local f="$1"
  local n="$2"
  [[ -f "$f" ]] && [[ "$(count_lines "$f")" -eq "$n" ]]
}

echo "[config] PROMPTS=$PROMPTS (n=$N)"
echo "[config] OUTDIR=$OUTDIR"
echo "[config] SYS_PROMPT=$SYS_PROMPT"
echo "[config] MAX_NEW_TOKENS=$MAX_NEW_TOKENS"

MODEL_KEYS=("llama3_1_8b_instruct" "gemma_7b_it" "mistral_7b_instruct" "qwen2_5_7b_instruct")
MODEL_IDS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "google/gemma-7b-it" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B-Instruct")

for gpu in 0 1 2 3; do
  (
    key="${MODEL_KEYS[$gpu]}"
    mid="${MODEL_IDS[$gpu]}"
    out="${OUTDIR}/${key}.jsonl"

    echo "[gpu $gpu] ${key} -> $out"

    if is_complete_jsonl "$out" "$N"; then
      echo "[gpu $gpu][skip] already complete: $out"
      exit 0
    fi

    CUDA_VISIBLE_DEVICES="$gpu" python scripts/lre_run_greedy_eval.py \
      --model_id "$mid" \
      --model_key "$key" \
      --prompts_jsonl "$PROMPTS" \
      --out_jsonl "$out" \
      --system_prompt "$SYS_PROMPT" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      2>&1 | tee "${LOGDIR}/${key}.log"

    echo "[gpu $gpu] done: $key"
  ) &
done

wait
echo "[done] generation finished: $OUTDIR"
