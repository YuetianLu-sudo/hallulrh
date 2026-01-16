#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override via inline env, no need to export globally) ----
RUN_TAG="${RUN_TAG:-chat_neutral_$(date +%Y%m%d_%H%M%S)}"
PROMPT_MODE="${PROMPT_MODE:-chat}"
SYS_PROMPT="${SYS_PROMPT:-You are a helpful assistant. Answer with a single short phrase.}"

FUTURE_PROMPTS="${FUTURE_PROMPTS:-experiments/sanity_check_1/eval/eval_prompts.jsonl}"
RELPANEL_PROMPTS="${RELPANEL_PROMPTS:-experiments/sanity_check_relpanel_ext6_q/eval/eval_prompts.jsonl}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

OUT_EVAL_FUTURE="${OUT_EVAL_FUTURE:-data/eval/future_${RUN_TAG}}"
OUT_JUDGE_FUTURE="${OUT_JUDGE_FUTURE:-data/judge_inputs/future_${RUN_TAG}}"
OUT_EVAL_RELPANEL="${OUT_EVAL_RELPANEL:-data/eval/relpanel_q_${RUN_TAG}}"
OUT_JUDGE_RELPANEL="${OUT_JUDGE_RELPANEL:-data/judge_inputs/relpanel_q_${RUN_TAG}}"

LOG_DIR="${LOG_DIR:-logs/eval_chat_${RUN_TAG}}"

# ---- Sanity checks ----
[[ -f "$FUTURE_PROMPTS" ]] || { echo "[error] FUTURE_PROMPTS not found: $FUTURE_PROMPTS" >&2; exit 1; }
[[ -f "$RELPANEL_PROMPTS" ]] || { echo "[error] RELPANEL_PROMPTS not found: $RELPANEL_PROMPTS" >&2; exit 1; }

mkdir -p "$OUT_EVAL_FUTURE" "$OUT_JUDGE_FUTURE" "$OUT_EVAL_RELPANEL" "$OUT_JUDGE_RELPANEL" "$LOG_DIR"

count_lines() { wc -l "$1" | awk '{print $1}'; }
N_FUTURE=$(count_lines "$FUTURE_PROMPTS")
N_RELPANEL=$(count_lines "$RELPANEL_PROMPTS")

is_complete_jsonl() {
  local f="$1"
  local n="$2"
  [[ -f "$f" ]] && [[ "$(count_lines "$f")" -eq "$n" ]]
}

echo "[config] RUN_TAG=$RUN_TAG"
echo "[config] PROMPT_MODE=$PROMPT_MODE"
echo "[config] SYS_PROMPT=$SYS_PROMPT"
echo "[config] FUTURE_PROMPTS=$FUTURE_PROMPTS (n=$N_FUTURE)"
echo "[config] RELPANEL_PROMPTS=$RELPANEL_PROMPTS (n=$N_RELPANEL)"
echo "[config] OUT_EVAL_FUTURE=$OUT_EVAL_FUTURE"
echo "[config] OUT_JUDGE_FUTURE=$OUT_JUDGE_FUTURE"
echo "[config] OUT_EVAL_RELPANEL=$OUT_EVAL_RELPANEL"
echo "[config] OUT_JUDGE_RELPANEL=$OUT_JUDGE_RELPANEL"
echo "[config] LOG_DIR=$LOG_DIR"

MODEL_KEYS=("llama3_1_8b_instruct" "gemma_7b_it" "mistral_7b_instruct" "qwen2_5_7b_instruct")
MODEL_IDS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "google/gemma-7b-it" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B-Instruct")

for gpu in 0 1 2 3; do
  (
    key="${MODEL_KEYS[$gpu]}"
    mid="${MODEL_IDS[$gpu]}"

    echo "[gpu $gpu] model_key=$key model_id=$mid"

    # ---- future ----
    fut_jsonl="$OUT_EVAL_FUTURE/${key}_baseline.jsonl"
    fut_csv="$OUT_JUDGE_FUTURE/${key}_baseline_for_judge.csv"
    if is_complete_jsonl "$fut_jsonl" "$N_FUTURE"; then
      echo "[skip] $key future already complete: $fut_jsonl"
    else
      echo "[run]  $key future -> $fut_jsonl"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_eval_from_prompts_for_judge.py \
        --model-id "$mid" \
        --model-name "${key}_baseline" \
        --prompts "$FUTURE_PROMPTS" \
        --jsonl-out "$fut_jsonl" \
        --csv-out "$fut_csv" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --prompt-mode "$PROMPT_MODE" \
        --system-prompt "$SYS_PROMPT" \
        > "${LOG_DIR}/${key}.future.gpu${gpu}.log" 2>&1
    fi

    # ---- relpanel_q ----
    rel_jsonl="$OUT_EVAL_RELPANEL/${key}_relpanel.jsonl"
    rel_csv="$OUT_JUDGE_RELPANEL/${key}_relpanel_for_judge.csv"
    if is_complete_jsonl "$rel_jsonl" "$N_RELPANEL"; then
      echo "[skip] $key relpanel_q already complete: $rel_jsonl"
    else
      echo "[run]  $key relpanel_q -> $rel_jsonl"
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_eval_from_prompts_for_judge.py \
        --model-id "$mid" \
        --model-name "${key}_relpanel" \
        --prompts "$RELPANEL_PROMPTS" \
        --jsonl-out "$rel_jsonl" \
        --csv-out "$rel_csv" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --prompt-mode "$PROMPT_MODE" \
        --system-prompt "$SYS_PROMPT" \
        > "${LOG_DIR}/${key}.relpanel_q.gpu${gpu}.log" 2>&1
    fi

    echo "[gpu $gpu] done: $key"
  ) &
done

wait
echo "[done] all eval_chat jobs finished."
