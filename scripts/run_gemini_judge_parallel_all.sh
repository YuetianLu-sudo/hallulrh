#!/usr/bin/env bash
set -euo pipefail

# Concurrency across *all* CSVs (future + relpanel). Gemini judge does not use GPU.
N_JOBS="${N_JOBS:-4}"

FUTURE_DIR="${FUTURE_DIR:-data/judge_inputs/future_chat_neutral_full_20251224_010241}"
RELPANEL_DIR="${RELPANEL_DIR:-data/judge_inputs/relpanel_q_chat_neutral_full_20251224_010241}"
LOG_DIR="${LOG_DIR:-logs/judge_all}"

mkdir -p "$LOG_DIR"

# Basic key check (fail fast)
: "${GEMINI_API_KEY:?need GEMINI_API_KEY (or GOOGLE_API_KEY)}"

inputs=()
shopt -s nullglob
for f in "$FUTURE_DIR"/*_for_judge.csv; do inputs+=("$f"); done
for f in "$RELPANEL_DIR"/*_for_judge.csv; do inputs+=("$f"); done
shopt -u nullglob

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "[error] No *_for_judge.csv found in:"
  echo "  - $FUTURE_DIR"
  echo "  - $RELPANEL_DIR"
  exit 1
fi

echo "[info] FUTURE_DIR=$FUTURE_DIR"
echo "[info] RELPANEL_DIR=$RELPANEL_DIR"
echo "[info] LOG_DIR=$LOG_DIR"
echo "[info] N_JOBS=$N_JOBS"
echo "[info] Found ${#inputs[@]} input CSV(s)."

pids=()
names=()
logs=()

for in_file in "${inputs[@]}"; do
  base="$(basename "$in_file")"
  split="$(basename "$(dirname "$in_file")")"
  out_file="${in_file/_for_judge.csv/_with_judge.csv}"
  log_file="$LOG_DIR/${split}.${base/_for_judge.csv/.log}"

  echo "[launch] $split/$base -> $(basename "$out_file")"
  echo "        log: $log_file"

  python -u scripts/lm_judge_gemini.py judge-csv \
    --input "$in_file" \
    --output "$out_file" \
    --resume \
    >"$log_file" 2>&1 &

  pids+=("$!")
  names+=("$split/$base")
  logs+=("$log_file")

  while [[ $(jobs -rp | wc -l | tr -d ' ') -ge "$N_JOBS" ]]; do
    sleep 1
  done
done

fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  log="${logs[$i]}"
  if wait "$pid"; then
    echo "[done] $name (pid=$pid)"
  else
    echo "[fail] $name (pid=$pid). See log: $log" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[error] At least one judge job failed. Re-run the same command; resume will continue." >&2
  exit 2
fi

echo "[all done] Gemini judge completed for all CSVs."
