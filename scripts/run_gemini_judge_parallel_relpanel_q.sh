#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${IN_DIR:-data/judge/relpanel_q}"
OUT_DIR="${OUT_DIR:-data/judge/relpanel_q}"
LOG_DIR="${LOG_DIR:-logs/judge_relpanel_q}"
N_JOBS="${N_JOBS:-4}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

shopt -s nullglob
inputs=("$IN_DIR"/*_for_judge.csv)
shopt -u nullglob

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "[error] No *_for_judge.csv found in: $IN_DIR" >&2
  exit 1
fi

echo "[info] IN_DIR=$IN_DIR"
echo "[info] OUT_DIR=$OUT_DIR"
echo "[info] LOG_DIR=$LOG_DIR"
echo "[info] N_JOBS=$N_JOBS"
echo "[info] Found ${#inputs[@]} input CSV(s)."

pids=()
names=()
logs=()

# Launch up to N_JOBS in parallel.
# Since relpanel_q has exactly 4 CSVs, this will run 4 judges concurrently.
for in_file in "${inputs[@]}"; do
  base="$(basename "$in_file")"
  out_file="$OUT_DIR/${base/_for_judge.csv/_with_judge.csv}"
  log_file="$LOG_DIR/${base/_for_judge.csv/.log}"

  if [[ -s "$out_file" ]]; then
    echo "[skip] already exists and non-empty: $out_file"
    continue
  fi

  echo "[launch] $base -> $(basename "$out_file")"
  echo "        log: $log_file"

  # -u: unbuffered stdout so progress appears in log immediately
  python -u scripts/lm_judge_gemini.py judge-csv \
    --input "$in_file" \
    --output "$out_file" \
    >"$log_file" 2>&1 &

  pids+=("$!")
  names+=("$base")
  logs+=("$log_file")

  # If you ever have >4 files, this keeps concurrency capped.
  while [[ $(jobs -rp | wc -l | tr -d ' ') -ge "$N_JOBS" ]]; do
    sleep 1
  done
done

# Wait and fail fast if any job fails.
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
  echo "[error] At least one judge job failed. Re-run this script after fixing the issue." >&2
  exit 2
fi

echo "[all done] Gemini judge completed for relpanel_q."
