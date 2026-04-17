#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NAT_EXP="runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209"
SYN_EXP="runs/experiments/exp1_lre21_synth_20260119_211221"
OUT="runs/experiments/lre_layer_prompt_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

declare -A MODEL_ID
MODEL_ID[gemma_7b_it]="google/gemma-7b-it"
MODEL_ID[llama3_1_8b_instruct]="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID[mistral_7b_instruct]="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ID[qwen2_5_7b_instruct]="Qwen/Qwen2.5-7B-Instruct"

declare -A PROMPTS
PROMPTS[gemma_7b_it]="$NAT_EXP/inputs/gemma_7b_it.for_3way_judge..with_gold.jsonl"
PROMPTS[llama3_1_8b_instruct]="$NAT_EXP/inputs/llama3_1_8b_instruct.for_3way_judge..with_gold.jsonl"
PROMPTS[mistral_7b_instruct]="$NAT_EXP/inputs/mistral_7b_instruct.for_3way_judge..with_gold.jsonl"
PROMPTS[qwen2_5_7b_instruct]="$NAT_EXP/inputs/qwen2_5_7b_instruct.for_3way_judge..with_gold.jsonl"

RELSET="$NAT_EXP/inputs/relation_set_all47.txt"

python - <<'PY'
from pathlib import Path
required = [
    "runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209/inputs/relation_set_all47.txt",
    "runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209/analysis_all47/lre3way_behavior_plus_deltacos_all47.csv",
    "runs/experiments/exp1_lre21_synth_20260119_211221/analysis/exp1_behavior_plus_deltacos_factual.csv",
]
for p in required:
    if not Path(p).exists():
        raise FileNotFoundError(p)
print("[ok] required inputs found")
PY

get_layers () {
  python - "$1" <<'PY'
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
}

run_step5 () {
  local variant="$1"
  local model_key="$2"
  local model_id="$3"
  local prompts_jsonl="$4"
  local extra_args="$5"

  local out_dir="$OUT/$variant/$model_key"
  mkdir -p "$out_dir"

  python scripts/lre_step5_deltacos_gold_per_triple.py \
    --model_id "$model_id" \
    --model_key "$model_key" \
    --prompts_jsonl "$prompts_jsonl" \
    --out_dir "$out_dir" \
    --train_frac 0.75 \
    --seed 0 \
    --batch_size 8 \
    $extra_args

  test -f "$out_dir/relation_summary.csv.gz"
}

for mk in gemma_7b_it llama3_1_8b_instruct mistral_7b_instruct qwen2_5_7b_instruct; do
  echo
  echo "===================="
  echo "[model] $mk"
  echo "===================="

  MID="${MODEL_ID[$mk]}"
  PJSON="${PROMPTS[$mk]}"

  if [ ! -f "$PJSON" ]; then
    echo "[FAIL] missing prompts jsonl: $PJSON"
    exit 1
  fi

  read -r L S_DEF O_DEF S_SHALLOW O_SHALLOW < <(get_layers "$MID")
  echo "[layers] n_layers=$L default=($S_DEF,$O_DEF) subj_shallow=($S_SHALLOW,$O_DEF) obj_shallow=($S_DEF,$O_SHALLOW)"

  run_step5 "default_chat"      "$mk" "$MID" "$PJSON" "--use_chat_template --subject_layer $S_DEF --object_layer $O_DEF"
  run_step5 "subj_shallow_chat" "$mk" "$MID" "$PJSON" "--use_chat_template --subject_layer $S_SHALLOW --object_layer $O_DEF"
  run_step5 "obj_shallow_chat"  "$mk" "$MID" "$PJSON" "--use_chat_template --subject_layer $S_DEF --object_layer $O_SHALLOW"
  run_step5 "default_plain"     "$mk" "$MID" "$PJSON" "--subject_layer $S_DEF --object_layer $O_DEF"
done

python scripts/lre_step5_compare_chat_vs_plain.py \
  --chat_base "$OUT/default_chat" \
  --plain_base "$OUT/default_plain" \
  --out_csv "$OUT/chat_vs_plain_compare.csv"

cat > "$OUT/summarize_layer_prompt_sensitivity.py" <<'PY'
import gzip
import math
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = Path("REPLACE_OUT")
NAT_BEHAV = Path("runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209/analysis_all47/lre3way_behavior_plus_deltacos_all47.csv")
SYN_BEHAV = Path("runs/experiments/exp1_lre21_synth_20260119_211221/analysis/exp1_behavior_plus_deltacos_factual.csv")

VARIANTS = ["default_chat", "subj_shallow_chat", "obj_shallow_chat", "default_plain"]
BASELINE = "default_chat"
MODEL_KEYS = ["gemma_7b_it", "llama3_1_8b_instruct", "mistral_7b_instruct", "qwen2_5_7b_instruct"]

def read_relsum(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)
    out = df.copy()
    if "relation_key" not in out.columns:
        raise KeyError(f"relation_key missing in {path}")
    if "delta_cos_mean_test" not in out.columns:
        raise KeyError(f"delta_cos_mean_test missing in {path}")
    out["delta_cos"] = out["delta_cos_mean_test"]
    return out

def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of {names} in columns: {list(df.columns)}")

nat = pd.read_csv(NAT_BEHAV)
nat_model_col = first_existing(nat, ["model_key", "model_name"])
nat_rel_col = first_existing(nat, ["relation_key", "relation"])
nat_y_col = first_existing(nat, ["hall_rate_answered", "hall_answered_rate"])

syn = pd.read_csv(SYN_BEHAV)
syn_model_col = first_existing(syn, ["model_name", "model_key"])
syn_rel_col = first_existing(syn, ["relation", "relation_key"])
syn_y_col = first_existing(syn, ["hall_rate"])

rows_rank = []
rows_nat = []
rows_syn = []

for mk in MODEL_KEYS:
    base = read_relsum(OUT / BASELINE / mk / "relation_summary.csv.gz")
    base = base[base["n_test"] > 10][["relation_key", "delta_cos"]].rename(columns={"delta_cos": "delta_cos_base"})

    nat_m = nat[nat[nat_model_col] == mk].copy()
    nat_m = nat_m[nat_m["n_test"] > 10][[nat_rel_col, nat_y_col]].rename(columns={nat_rel_col: "relation_key", nat_y_col: "hall_rate_answered"})

    syn_m = syn[syn[syn_model_col] == mk].copy()
    syn_m = syn_m[[syn_rel_col, syn_y_col]].rename(columns={syn_rel_col: "relation_key", syn_y_col: "hall_rate"})

    for variant in VARIANTS:
        alt = read_relsum(OUT / variant / mk / "relation_summary.csv.gz")
        alt_n = alt[alt["n_test"] > 10][["relation_key", "delta_cos"]].rename(columns={"delta_cos": "delta_cos_alt"})

        m = base.merge(alt_n, on="relation_key", how="inner")
        r_rank, p_rank = pearsonr(m["delta_cos_base"], m["delta_cos_alt"])
        rho_rank, p_s_rank = spearmanr(m["delta_cos_base"], m["delta_cos_alt"])
        rows_rank.append({
            "model_key": mk,
            "variant": variant,
            "n_rel": len(m),
            "pearson_vs_default": r_rank,
            "pearson_p": p_rank,
            "spearman_vs_default": rho_rank,
            "spearman_p": p_s_rank,
            "mean_abs_delta": (m["delta_cos_alt"] - m["delta_cos_base"]).abs().mean(),
        })

        mn = alt_n.merge(nat_m, on="relation_key", how="inner")
        r_nat, p_nat = pearsonr(mn["delta_cos_alt"], mn["hall_rate_answered"])
        rho_nat, p_s_nat = spearmanr(mn["delta_cos_alt"], mn["hall_rate_answered"])
        rows_nat.append({
            "model_key": mk,
            "variant": variant,
            "n_rel": len(mn),
            "pearson_nat": r_nat,
            "pearson_nat_p": p_nat,
            "spearman_nat": rho_nat,
            "spearman_nat_p": p_s_nat,
        })

        ms = alt[["relation_key", "delta_cos"]].rename(columns={"delta_cos": "delta_cos_alt"}).merge(syn_m, on="relation_key", how="inner")
        r_syn, p_syn = pearsonr(ms["delta_cos_alt"], ms["hall_rate"])
        rho_syn, p_s_syn = spearmanr(ms["delta_cos_alt"], ms["hall_rate"])
        rows_syn.append({
            "model_key": mk,
            "variant": variant,
            "n_rel": len(ms),
            "pearson_syn": r_syn,
            "pearson_syn_p": p_syn,
            "spearman_syn": rho_syn,
            "spearman_syn_p": p_s_syn,
        })

rank_df = pd.DataFrame(rows_rank)
nat_df = pd.DataFrame(rows_nat)
syn_df = pd.DataFrame(rows_syn)

rank_df.to_csv(OUT / "layer_prompt_rank_stability.csv", index=False)
nat_df.to_csv(OUT / "layer_prompt_natural_behavior_corr.csv", index=False)
syn_df.to_csv(OUT / "layer_prompt_synthetic_behavior_corr.csv", index=False)

print("[write]", OUT / "chat_vs_plain_compare.csv")
print("[write]", OUT / "layer_prompt_rank_stability.csv")
print("[write]", OUT / "layer_prompt_natural_behavior_corr.csv")
print("[write]", OUT / "layer_prompt_synthetic_behavior_corr.csv")
print()
print("=== Rank stability vs default_chat ===")
print(rank_df.to_string(index=False))
print()
print("=== Natural Figure-2 correlations ===")
print(nat_df.to_string(index=False))
print()
print("=== Synthetic Figure-1 correlations ===")
print(syn_df.to_string(index=False))
PY

python - <<PY
from pathlib import Path
p = Path("$OUT/summarize_layer_prompt_sensitivity.py")
s = p.read_text(encoding="utf-8")
s = s.replace("REPLACE_OUT", "$OUT")
p.write_text(s, encoding="utf-8")
print("[ok] wrote summary script:", p)
PY

python "$OUT/summarize_layer_prompt_sensitivity.py"

echo
echo "[done] Inspect these files:"
echo "  $OUT/chat_vs_plain_compare.csv"
echo "  $OUT/layer_prompt_rank_stability.csv"
echo "  $OUT/layer_prompt_natural_behavior_corr.csv"
echo "  $OUT/layer_prompt_synthetic_behavior_corr.csv"
