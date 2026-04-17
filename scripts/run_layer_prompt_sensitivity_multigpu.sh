#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mounts/work/yuetian_lu/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TMPDIR="${TMPDIR:-/mounts/work/yuetian_lu/tmp}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$TMPDIR"

OUT="runs/experiments/lre_layer_prompt_sensitivity_multigpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/logs"
echo "$OUT" > runs/layer_prompt_sensitivity_multigpu.latest

mapfile -t FREE_GPUS < <(
python - <<'PY'
import subprocess

cmd = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]
txt = subprocess.check_output(cmd, text=True)
rows = []
for line in txt.strip().splitlines():
    idx, mem, util = [x.strip() for x in line.split(",")]
    idx = int(idx); mem = int(mem); util = int(util)
    if mem <= 1024 and util <= 10:
        rows.append((idx, mem, util))
rows.sort(key=lambda x: (x[1], x[2], x[0]))
for idx, _, _ in rows:
    print(idx)
PY
)

if [ "${#FREE_GPUS[@]}" -lt 4 ]; then
  echo "[FAIL] need 4 near-idle GPUs, found ${#FREE_GPUS[@]}: ${FREE_GPUS[*]:-none}"
  exit 1
fi

GPU_A="${FREE_GPUS[0]}"
GPU_B="${FREE_GPUS[1]}"
GPU_C="${FREE_GPUS[2]}"
GPU_D="${FREE_GPUS[3]}"

echo "[OUT] $OUT"
echo "[GPU MAP] gemma_7b_it -> $GPU_A"
echo "[GPU MAP] llama3_1_8b_instruct -> $GPU_B"
echo "[GPU MAP] mistral_7b_instruct -> $GPU_C"
echo "[GPU MAP] qwen2_5_7b_instruct -> $GPU_D"

bash scripts/run_layer_prompt_sensitivity_one_model.sh gemma_7b_it            "$GPU_A" "$OUT" > "$OUT/logs/gemma_7b_it.log" 2>&1 &
PID1=$!
bash scripts/run_layer_prompt_sensitivity_one_model.sh llama3_1_8b_instruct   "$GPU_B" "$OUT" > "$OUT/logs/llama3_1_8b_instruct.log" 2>&1 &
PID2=$!
bash scripts/run_layer_prompt_sensitivity_one_model.sh mistral_7b_instruct    "$GPU_C" "$OUT" > "$OUT/logs/mistral_7b_instruct.log" 2>&1 &
PID3=$!
bash scripts/run_layer_prompt_sensitivity_one_model.sh qwen2_5_7b_instruct    "$GPU_D" "$OUT" > "$OUT/logs/qwen2_5_7b_instruct.log" 2>&1 &
PID4=$!

echo "$PID1" > "$OUT/logs/gemma_7b_it.pid"
echo "$PID2" > "$OUT/logs/llama3_1_8b_instruct.pid"
echo "$PID3" > "$OUT/logs/mistral_7b_instruct.pid"
echo "$PID4" > "$OUT/logs/qwen2_5_7b_instruct.pid"

FAIL=0
wait "$PID1" || FAIL=1
wait "$PID2" || FAIL=1
wait "$PID3" || FAIL=1
wait "$PID4" || FAIL=1

if [ "$FAIL" -ne 0 ]; then
  echo "[FAIL] at least one model worker failed"
  exit 1
fi

python scripts/lre_step5_compare_chat_vs_plain.py \
  --chat_base "$OUT/default_chat" \
  --plain_base "$OUT/default_plain" \
  --out_csv "$OUT/chat_vs_plain_compare.csv"

cat > "$OUT/summarize_layer_prompt_sensitivity.py" <<'PY'
import gzip
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
s = p.read_text(encoding="utf-8").replace("REPLACE_OUT", "$OUT")
p.write_text(s, encoding="utf-8")
print("[ok] wrote summary script:", p)
PY

python "$OUT/summarize_layer_prompt_sensitivity.py"

echo
echo "[done] OUT=$OUT"
echo "[done] inspect:"
echo "  $OUT/chat_vs_plain_compare.csv"
echo "  $OUT/layer_prompt_rank_stability.csv"
echo "  $OUT/layer_prompt_natural_behavior_corr.csv"
echo "  $OUT/layer_prompt_synthetic_behavior_corr.csv"
