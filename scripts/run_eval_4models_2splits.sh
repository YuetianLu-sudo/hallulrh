#!/usr/bin/env bash
set -euo pipefail

FUTURE_PROMPTS="experiments/sanity_check_1/eval/eval_prompts.jsonl"
RELPANEL_PROMPTS="experiments/sanity_check_relpanel_ext6/eval/eval_prompts.jsonl"

# Format: model_key|hf_model_id|gpu_id
MODELS=(
  "llama3_1_8b_instruct|meta-llama/Meta-Llama-3.1-8B-Instruct|0"
  "gemma_7b_it|google/gemma-7b-it|1"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3|2"
  "qwen2_5_7b_instruct|Qwen/Qwen2.5-7B-Instruct|3"
)

MAX_NEW_TOKENS=128

run_one() {
  local model_key="$1"
  local model_id="$2"
  local gpu="$3"

  export CUDA_VISIBLE_DEVICES="$gpu"

  echo "[gpu ${gpu}] model_key=${model_key} model_id=${model_id}"

  # --- future split (baseline prompts, reused) ---
  python scripts/run_eval_from_prompts_for_judge.py \
    --model-id   "$model_id" \
    --model-name "${model_key}_baseline" \
    --prompts    "$FUTURE_PROMPTS" \
    --jsonl-out  "data/eval/future/${model_key}_baseline.jsonl" \
    --csv-out    "data/judge/future/${model_key}_baseline_for_judge.csv" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  # --- relpanel split (extended prompts, new file) ---
  python scripts/run_eval_from_prompts_for_judge.py \
    --model-id   "$model_id" \
    --model-name "${model_key}_relpanel" \
    --prompts    "$RELPANEL_PROMPTS" \
    --jsonl-out  "data/eval/relpanel/${model_key}_relpanel.jsonl" \
    --csv-out    "data/judge/relpanel/${model_key}_relpanel_for_judge.csv" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  echo "[gpu ${gpu}] done: ${model_key}"
}

pids=()
for entry in "${MODELS[@]}"; do
  IFS="|" read -r key mid gpu <<<"$entry"
  run_one "$key" "$mid" "$gpu" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[done] All eval runs finished."
