#!/usr/bin/env bash
set -euo pipefail

# English comments only.

RELPANEL_PROMPTS_Q="${RELPANEL_PROMPTS_Q:-experiments/sanity_check_relpanel_ext6_q/eval/eval_prompts.jsonl}"
OUT_EVAL_DIR="${OUT_EVAL_DIR:-data/eval/relpanel_q}"
OUT_JUDGE_DIR="${OUT_JUDGE_DIR:-data/judge/relpanel_q}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

# Which GPUs to watch/use
GPU_IDS=(${GPU_IDS:-0 1 2 3})

# Consider a GPU "free" only if:
# 1) no compute processes are running on it, AND
# 2) used memory is <= GPU_USED_MAX_MIB, AND
# 3) utilization is <= GPU_UTIL_MAX
GPU_USED_MAX_MIB="${GPU_USED_MAX_MIB:-2000}"
GPU_UTIL_MAX="${GPU_UTIL_MAX:-10}"

# Poll interval (seconds)
POLL_SECS="${POLL_SECS:-30}"

mkdir -p "$OUT_EVAL_DIR" "$OUT_JUDGE_DIR" logs/relpanel_q

# One job per model. Format: "model_key|hf_model_id"
MODELS=(
  "llama3_1_8b_instruct|meta-llama/Meta-Llama-3.1-8B-Instruct"
  "gemma_7b_it|google/gemma-7b-it"
  "mistral_7b_instruct|mistralai/Mistral-7B-Instruct-v0.3"
  "qwen2_5_7b_instruct|Qwen/Qwen2.5-7B-Instruct"
)

if [[ ! -f "$RELPANEL_PROMPTS_Q" ]]; then
  echo "[error] RELPANEL_PROMPTS_Q not found: $RELPANEL_PROMPTS_Q" >&2
  exit 1
fi

expected_lines="$(wc -l < "$RELPANEL_PROMPTS_Q" | tr -d ' ')"

# Clean stale lock files (same user)
for g in "${GPU_IDS[@]}"; do
  lock="/tmp/hallulrh_${USER}_gpu_lock_${g}"
  if [[ -f "$lock" ]]; then
    pid="$(cat "$lock" 2>/dev/null || true)"
    if [[ -z "${pid}" ]] || ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$lock"
    fi
  fi
done

gpu_is_free() {
  local g="$1"
  local lock="/tmp/hallulrh_${USER}_gpu_lock_${g}"
  [[ -f "$lock" ]] && return 1

  # If any compute process exists on that GPU, treat as busy.
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$g" | grep -qE '^[0-9]+'; then
    return 1
  fi

  local used util
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | head -n1 | tr -d ' ')"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | head -n1 | tr -d ' ')"

  [[ -z "$used" || -z "$util" ]] && return 1
  (( used <= GPU_USED_MAX_MIB )) && (( util <= GPU_UTIL_MAX ))
}

launch_job() {
  local g="$1"
  local model_key="$2"
  local model_id="$3"

  local lock="/tmp/hallulrh_${USER}_gpu_lock_${g}"
  local log="logs/relpanel_q/${model_key}.gpu${g}.log"

  # Outputs (directory disambiguates cloze vs question)
  local out_jsonl="${OUT_EVAL_DIR}/${model_key}_relpanel.jsonl"
  local out_csv="${OUT_JUDGE_DIR}/${model_key}_relpanel_for_judge.csv"

  echo $$ > "$lock"

  (
    set -euo pipefail
    trap 'rm -f "'"$lock"'"' EXIT
    export CUDA_VISIBLE_DEVICES="$g"
    export PYTHONUNBUFFERED=1

    echo "[start] $(date) gpu=$g model_key=$model_key model_id=$model_id"
    echo "[info] prompts=$RELPANEL_PROMPTS_Q expected_lines=$expected_lines max_new_tokens=$MAX_NEW_TOKENS"
    python scripts/run_eval_from_prompts_for_judge.py \
      --model-id "$model_id" \
      --model-name "${model_key}_relpanel" \
      --prompts "$RELPANEL_PROMPTS_Q" \
      --jsonl-out "$out_jsonl" \
      --csv-out "$out_csv" \
      --max-new-tokens "$MAX_NEW_TOKENS"
    echo "[done] $(date) gpu=$g model_key=$model_key"
  ) >"$log" 2>&1 &

  echo $!
}

# Build pending list (skip completed)
declare -a pending=()
for m in "${MODELS[@]}"; do
  key="${m%%|*}"
  out_jsonl="${OUT_EVAL_DIR}/${key}_relpanel.jsonl"
  if [[ -f "$out_jsonl" ]]; then
    n="$(wc -l < "$out_jsonl" | tr -d ' ')"
    if [[ "$n" == "$expected_lines" ]]; then
      echo "[skip] ${key} already complete: $out_jsonl ($n/$expected_lines)"
      continue
    else
      echo "[redo] ${key} incomplete: $out_jsonl ($n/$expected_lines)"
    fi
  fi
  pending+=("$m")
done

declare -A running_pid=()  # gpu -> pid
declare -A running_key=()  # gpu -> model_key

echo "[queue] pending jobs: ${#pending[@]} (expected_lines=$expected_lines)"
echo "[queue] watching GPUs: ${GPU_IDS[*]} (free: no compute procs, used<=${GPU_USED_MAX_MIB}MiB, util<=${GPU_UTIL_MAX}%)"

while (( ${#pending[@]} > 0 )) || (( ${#running_pid[@]} > 0 )); do
  # Reap finished jobs
  for g in "${!running_pid[@]}"; do
    pid="${running_pid[$g]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[reap] gpu=$g finished model=${running_key[$g]} pid=$pid"
      unset 'running_pid[$g]'
      unset 'running_key[$g]'
    fi
  done

  # Launch on any newly free GPU
  for g in "${GPU_IDS[@]}"; do
    [[ -n "${running_pid[$g]:-}" ]] && continue
    (( ${#pending[@]} == 0 )) && break

    if gpu_is_free "$g"; then
      m="${pending[0]}"
      pending=("${pending[@]:1}")

      key="${m%%|*}"
      mid="${m#*|}"

      echo "[launch] gpu=$g model=$key"
      pid="$(launch_job "$g" "$key" "$mid")"
      running_pid["$g"]="$pid"
      running_key["$g"]="$key"
      echo "[launch] gpu=$g pid=$pid log=logs/relpanel_q/${key}.gpu${g}.log"
    fi
  done

  if (( ${#pending[@]} > 0 )); then
    echo "[wait] pending=${#pending[@]} running=${#running_pid[@]} (sleep ${POLL_SECS}s)"
    sleep "$POLL_SECS"
  fi
done

echo "[all done] $(date)"
