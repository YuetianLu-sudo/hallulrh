#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${1:?need MODEL_KEY}"
GPU_ID="${2:?need GPU_ID}"
OUT="${3:?need OUT}"

export HF_HOME="${HF_HOME:-/mounts/work/yuetian_lu/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TMPDIR="${TMPDIR:-/mounts/work/yuetian_lu/tmp}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$TMPDIR"
mkdir -p "$OUT"

NAT_EXP="runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209"

case "$MODEL_KEY" in
  gemma_7b_it)
    MODEL_ID="google/gemma-7b-it"
    PJSON="$NAT_EXP/inputs/gemma_7b_it.for_3way_judge..with_gold.jsonl"
    ;;
  llama3_1_8b_instruct)
    MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    PJSON="$NAT_EXP/inputs/llama3_1_8b_instruct.for_3way_judge..with_gold.jsonl"
    ;;
  mistral_7b_instruct)
    MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
    PJSON="$NAT_EXP/inputs/mistral_7b_instruct.for_3way_judge..with_gold.jsonl"
    ;;
  qwen2_5_7b_instruct)
    MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
    PJSON="$NAT_EXP/inputs/qwen2_5_7b_instruct.for_3way_judge..with_gold.jsonl"
    ;;
  *)
    echo "[FAIL] unknown MODEL_KEY=$MODEL_KEY"
    exit 1
    ;;
esac

read -r L S_DEF O_DEF S_SHALLOW O_SHALLOW < <(
python - "$MODEL_ID" <<'PY'
import sys
from transformers import AutoConfig

model_id = sys.argv[1]
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
L = int(getattr(cfg, "num_hidden_layers"))
s_def = L // 2
o_def = max(s_def + 2, L - 2)
s_shallow = max(1, L // 3)
o_shallow = max(s_def + 2, L - 4)
print(L, s_def, o_def, s_shallow, o_shallow)
PY
)

echo "[model] $MODEL_KEY"
echo "[gpu]   $GPU_ID"
echo "[layers] n_layers=$L default=($S_DEF,$O_DEF) subj_shallow=($S_SHALLOW,$O_DEF) obj_shallow=($S_DEF,$O_SHALLOW)"

run_step5 () {
  local variant="$1"
  local extra_args="$2"
  local out_dir="$OUT/$variant/$MODEL_KEY"
  mkdir -p "$out_dir"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/lre_step5_deltacos_gold_per_triple.py \
    --model_id "$MODEL_ID" \
    --model_key "$MODEL_KEY" \
    --prompts_jsonl "$PJSON" \
    --out_dir "$out_dir" \
    --train_frac 0.75 \
    --seed 0 \
    --batch_size 8 \
    $extra_args

  test -f "$out_dir/relation_summary.csv.gz"
}

run_step5 "default_chat"      "--use_chat_template --subject_layer $S_DEF --object_layer $O_DEF"
run_step5 "subj_shallow_chat" "--use_chat_template --subject_layer $S_SHALLOW --object_layer $O_DEF"
run_step5 "obj_shallow_chat"  "--use_chat_template --subject_layer $S_DEF --object_layer $O_SHALLOW"
run_step5 "default_plain"     "--subject_layer $S_DEF --object_layer $O_DEF"

echo "[done] $MODEL_KEY"
