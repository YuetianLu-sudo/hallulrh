#!/usr/bin/env bash
set -euo pipefail

FUTURE_PROMPTS="experiments/sanity_check_1/eval/eval_prompts.jsonl"
RELPANEL_PROMPTS="experiments/sanity_check_relpanel_ext6/eval/eval_prompts.jsonl"

# Freeze decoding setting here (final paper setting)
MAX_NEW_TOKENS=128

# Format: model_key|hf_model_id|gpu_id
MODELS=(
  "llama3_1_8b_instruct|meta-llama/Meta-Llama-3.1-8B-Instruct|0"
  "gemma_7b_it|google/gemma-7b-it|1"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3|2"
  "qwen2_5_7b_instruct|Qwen/Qwen2.5-7B-Instruct|3"
)

expected_lines () {
  local f="$1"
  wc -l "$f" | awk '{print $1}'
}

is_complete () {
  local out_jsonl="$1"
  local n_expected="$2"
  if [[ ! -f "$out_jsonl" ]]; then
    return 1
  fi
  local n_out
  n_out=$(wc -l "$out_jsonl" | awk '{print $1}')
  [[ "$n_out" -eq "$n_expected" ]]
}

run_split () {
  local model_key="$1"
  local model_id="$2"
  local split="$3"          # future or relpanel
  local prompts="$4"
  local jsonl_out="$5"
  local csv_out="$6"

  local n_expected
  n_expected=$(expected_lines "$prompts")

  if is_complete "$jsonl_out" "$n_expected"; then
    echo "[skip] ${model_key} ${split} already complete: ${jsonl_out}"
    return 0
  fi

  echo "[run]  ${model_key} ${split} -> ${jsonl_out}"
  python scripts/run_eval_from_prompts_for_judge.py \
    --model-id   "$model_id" \
    --model-name "${model_key}_${split}" \
    --prompts    "$prompts" \
    --jsonl-out  "$jsonl_out" \
    --csv-out    "$csv_out" \
    --max-new-tokens "$MAX_NEW_TOKENS"
}

run_one () {
  local model_key="$1"
  local model_id="$2"
  local gpu="$3"

  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[gpu ${gpu}] model_key=${model_key} model_id=${model_id}"

  mkdir -p data/eval/future data/eval/relpanel data/judge/future data/judge/relpanel

  run_split "$model_key" "$model_id" "baseline" "$FUTURE_PROMPTS" \
    "data/eval/future/${model_key}_baseline.jsonl" \
    "data/judge/future/${model_key}_baseline_for_judge.csv"

  run_split "$model_key" "$model_id" "relpanel" "$RELPANEL_PROMPTS" \
    "data/eval/relpanel/${model_key}_relpanel.jsonl" \
    "data/judge/relpanel/${model_key}_relpanel_for_judge.csv"

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

echo "[done] All missing eval runs finished."
