import gzip
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

FIG2 = Path("runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209/analysis_all47/lre3way_behavior_plus_deltacos_all47.csv")

fig2 = pd.read_csv(FIG2)
if "model_key" not in fig2.columns and "model_name" in fig2.columns:
    fig2 = fig2.rename(columns={"model_name": "model_key"})
if "relation_key" not in fig2.columns and "relation" in fig2.columns:
    fig2 = fig2.rename(columns={"relation": "relation_key"})

fig2 = fig2[fig2["n_test"] > 10].copy()
fig2 = fig2[["model_key", "relation_key", "cos_improvement", "n_test"]].copy()

def read_summary(p: Path):
    try:
        if str(p).endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8") as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(p)
    except Exception:
        return None

    if "relation_key" not in df.columns:
        return None

    delta_col = None
    for c in ["cos_improvement", "delta_cos", "delta_cos_mean_test", "delta_cos_mean", "cos_improvement_mean"]:
        if c in df.columns:
            delta_col = c
            break
    if delta_col is None:
        return None

    if "model_key" not in df.columns:
        # Infer from path if possible.
        parts = p.parts
        mk = None
        for part in parts:
            if part in {"gemma_7b_it", "llama3_1_8b_instruct", "mistral_7b_instruct", "qwen2_5_7b_instruct"}:
                mk = part
        if mk is None:
            return None
        df["model_key"] = mk

    out = df[["model_key", "relation_key", delta_col]].copy()
    out = out.rename(columns={delta_col: "candidate_delta"})
    return out

rows = []

for p in Path("runs").rglob("relation_summary.csv.gz"):
    df = read_summary(p)
    if df is None:
        continue

    for mk, f in fig2.groupby("model_key"):
        d = df[df["model_key"].astype(str) == mk].copy()
        if d.empty:
            continue

        m = f.merge(d, on=["model_key", "relation_key"], how="inner")
        if len(m) < 5:
            continue

        r, rp = pearsonr(m["cos_improvement"], m["candidate_delta"])
        rho, sp = spearmanr(m["cos_improvement"], m["candidate_delta"])

        rows.append({
            "model_key": mk,
            "candidate_path": str(p),
            "candidate_model_dir": str(p.parent),
            "n_overlap_fig2": len(m),
            "pearson_vs_fig2": r,
            "spearman_vs_fig2": rho,
            "mean_abs_diff": (m["cos_improvement"] - m["candidate_delta"]).abs().mean(),
        })

res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit("[FAIL] no candidates found")

res = res.sort_values(
    ["model_key", "n_overlap_fig2", "pearson_vs_fig2", "mean_abs_diff"],
    ascending=[True, False, False, True],
)

out = Path("runs/fig2_step5_base_candidates.csv")
res.to_csv(out, index=False)
print("[write]", out)

print("\n=== best candidates per model ===")
best = res.groupby("model_key", as_index=False).head(10)
for mk, g in best.groupby("model_key"):
    print("\nMODEL:", mk)
    print(g[["n_overlap_fig2", "pearson_vs_fig2", "spearman_vs_fig2", "mean_abs_diff", "candidate_model_dir"]].head(10).to_string(index=False))
