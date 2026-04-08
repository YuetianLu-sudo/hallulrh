#!/usr/bin/env bash
set -euo pipefail

# ---------- user-configurable ----------
RUN="${RUN:-runs/experiments/natural_retrieval_$(date +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-data/lre_hernandez/prompts/lre_prompts_qonly.jsonl}"
BASE_ROOT="${BASE_ROOT:-data/lre_hernandez/deltacos/lre_step5_deltacos_gold_20260105_122321}"
RELSET="${RELSET:-runs/experiments/ablation_affine_20260217_023455/inputs/relset_fig2_min11_intersection.txt}"

METRIC="${METRIC:-cosine}"              # cosine (primary) or euclidean
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-train}"   # train (primary) or all
QUERY_MODE="${QUERY_MODE:-translated}"          # translated (primary) or subject_only
BATCH_SIZE="${BATCH_SIZE:-8}"

# "idle GPU" heuristic:
MAX_USED_MB="${MAX_USED_MB:-2000}"      # only use GPUs with <=2GB currently used
MAX_UTIL="${MAX_UTIL:-10}"              # and <=10% utilization
POLL_SEC="${POLL_SEC:-20}"              # how often to poll for free GPUs
MAX_PARALLEL="${MAX_PARALLEL:-4}"       # upper bound on simultaneous jobs
# --------------------------------------

mkdir -p "$RUN"

MODELS=(
  "gemma_7b_it"
  "llama3_1_8b_instruct"
  "mistral_7b_instruct"
  "qwen2_5_7b_instruct"
)

echo "[info] RUN=$RUN"
echo "[info] PROMPTS=$PROMPTS"
echo "[info] BASE_ROOT=$BASE_ROOT"
echo "[info] RELSET=$RELSET"
echo "[info] metric=$METRIC candidate_source=$CANDIDATE_SOURCE query_mode=$QUERY_MODE"
echo "[info] idle-gpu rule: memory.used <= $MAX_USED_MB MB and util <= $MAX_UTIL%"

declare -A PID2GPU=()
declare -A GPU2PID=()

refresh_finished_jobs() {
  local pid gpu
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu="${PID2GPU[$pid]}"
      wait "$pid" || true
      echo "[done] pid=$pid finished on GPU $gpu"
      unset PID2GPU["$pid"]
      unset GPU2PID["$gpu"]
    fi
  done
}

get_idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F',' -v max_used="$MAX_USED_MB" -v max_util="$MAX_UTIL" '
    {
      gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3);
      if (($2+0) <= max_used && ($3+0) <= max_util) print $1;
    }'
}

launch_one() {
  local mk="$1"
  local gpu="$2"
  local out="$RUN/$mk"
  mkdir -p "$out"

  echo "[launch] $mk on GPU $gpu -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_natural_lre_retrieval.py \
    --prompts "$PROMPTS" \
    --base-step5-dir "$BASE_ROOT/$mk" \
    --relation-set "$RELSET" \
    --outdir "$out" \
    --metric "$METRIC" \
    --candidate-source "$CANDIDATE_SOURCE" \
    --query-mode "$QUERY_MODE" \
    --batch-size "$BATCH_SIZE" \
    --use-chat-template \
    --system-prompt "You are a helpful assistant. Answer with a single short phrase." \
    > "$out/run.log" 2>&1 &

  local pid=$!
  PID2GPU["$pid"]="$gpu"
  GPU2PID["$gpu"]="$pid"
  echo "[launch] pid=$pid"
}

queue=("${MODELS[@]}")

while true; do
  refresh_finished_jobs

  running_count="${#PID2GPU[@]}"
  if [[ "${#queue[@]}" -eq 0 && "$running_count" -eq 0 ]]; then
    break
  fi

  if [[ "${#queue[@]}" -gt 0 && "$running_count" -lt "$MAX_PARALLEL" ]]; then
    mapfile -t idle_gpus < <(get_idle_gpus)

    if [[ "${#idle_gpus[@]}" -gt 0 ]]; then
      for gpu in "${idle_gpus[@]}"; do
        [[ "$running_count" -ge "$MAX_PARALLEL" ]] && break
        [[ "${#queue[@]}" -eq 0 ]] && break

        # skip GPUs already used by one of our jobs
        if [[ -n "${GPU2PID[$gpu]:-}" ]]; then
          continue
        fi

        mk="${queue[0]}"
        queue=("${queue[@]:1}")
        launch_one "$mk" "$gpu"
        running_count="${#PID2GPU[@]}"
      done
    fi
  fi

  if [[ "${#queue[@]}" -gt 0 || "${#PID2GPU[@]}" -gt 0 ]]; then
    sleep "$POLL_SEC"
  fi
done

echo "[done] all retrieval jobs finished."
echo "[done] outputs under: $RUN"
