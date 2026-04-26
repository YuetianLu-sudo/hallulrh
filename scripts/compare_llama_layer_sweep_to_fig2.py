import gzip
import os
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

variant = os.environ.get("LLAMA_SWEEP_PROMPT_VARIANT", "plain")
latest = Path(f"runs/llama_all47_layer_sweep_{variant}.latest")
if not latest.exists():
    raise FileNotFoundError(latest)

root = Path(latest.read_text(encoding="utf-8").strip())

fig2 = pd.read_csv("runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209/analysis_all47/lre3way_behavior_plus_deltacos_all47.csv")
fig2 = fig2[(fig2["model_key"] == "llama3_1_8b_instruct") & (fig2["n_test"] > 10)].copy()
fig2 = fig2[["relation_key", "cos_improvement"]]

rows = []

for p in sorted(root.glob("*/relation_summary.csv.gz")):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)

    if "cos_improvement" in df.columns:
        dcol = "cos_improvement"
    elif "delta_cos_mean_test" in df.columns:
        dcol = "delta_cos_mean_test"
    elif "delta_cos" in df.columns:
        dcol = "delta_cos"
    else:
        continue

    d = df[["relation_key", dcol]].rename(columns={dcol: "candidate_delta"})
    d = d.groupby("relation_key", as_index=False)["candidate_delta"].mean()
    m = fig2.merge(d, on="relation_key", how="inner")
    if len(m) < 10:
        continue

    r, rp = pearsonr(m["cos_improvement"], m["candidate_delta"])
    rho, sp = spearmanr(m["cos_improvement"], m["candidate_delta"])
    mad = (m["cos_improvement"] - m["candidate_delta"]).abs().mean()
    max_abs = (m["cos_improvement"] - m["candidate_delta"]).abs().max()

    rows.append({
        "variant": p.parent.name,
        "n_overlap_fig2": len(m),
        "pearson_vs_fig2": r,
        "spearman_vs_fig2": rho,
        "mean_abs_diff": mad,
        "max_abs_diff": max_abs,
        "base_dir": str(p.parent),
    })

res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit("[FAIL] no layer-sweep summaries found")

res = res.sort_values(["n_overlap_fig2", "pearson_vs_fig2", "mean_abs_diff"], ascending=[False, False, True])
out = Path(f"runs/llama_layer_sweep_{variant}_vs_fig2.csv")
res.to_csv(out, index=False)

print("[write]", out)
print()
print("=== top candidates ===")
print(res.head(25).to_string(index=False))

best = res.iloc[0]
print()
print("=== best ===")
print(best.to_string())

if best["n_overlap_fig2"] == 28 and best["pearson_vs_fig2"] > 0.999 and best["mean_abs_diff"] < 1e-3:
    print()
    print("[OK] Found Llama base matching Figure 2:")
    print(best["base_dir"])
else:
    print()
    print("[WARN] No Llama layer pair perfectly matches Figure 2.")
