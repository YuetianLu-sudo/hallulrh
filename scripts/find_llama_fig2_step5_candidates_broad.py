import gzip
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

FIG2 = Path("runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209/analysis_all47/lre3way_behavior_plus_deltacos_all47.csv")
MODEL = "llama3_1_8b_instruct"

fig2 = pd.read_csv(FIG2)
if "model_key" not in fig2.columns and "model_name" in fig2.columns:
    fig2 = fig2.rename(columns={"model_name": "model_key"})
if "relation_key" not in fig2.columns and "relation" in fig2.columns:
    fig2 = fig2.rename(columns={"relation": "relation_key"})

fig2 = fig2[(fig2["model_key"] == MODEL) & (fig2["n_test"] > 10)].copy()
fig2 = fig2[["relation_key", "cos_improvement"]].drop_duplicates()

delta_candidates = [
    "cos_improvement",
    "delta_cos",
    "delta_cos_mean",
    "delta_cos_mean_test",
    "cos_improvement_mean",
    "mean_delta_cos",
]

rows = []

def read_csv_any(p: Path):
    try:
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return pd.read_csv(f)
        return pd.read_csv(p)
    except Exception:
        return None

for p in Path("runs").rglob("*"):
    if not p.is_file():
        continue
    if not (str(p).endswith(".csv") or str(p).endswith(".csv.gz")):
        continue
    if p.stat().st_size > 250_000_000:
        continue

    # Prefer plausible summary files, but still allow broader search.
    name = p.name.lower()
    path_s = str(p).lower()
    if not any(x in name for x in ["summary", "relation", "deltacos", "delta", "lre", "behavior"]) and "llama" not in path_s:
        continue

    df = read_csv_any(p)
    if df is None or "relation_key" not in df.columns:
        continue

    dcol = None
    for c in delta_candidates:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        continue

    d = df.copy()

    # Determine whether this file contains model-specific rows.
    if "model_key" in d.columns:
        d = d[d["model_key"].astype(str) == MODEL]
    elif "model_name" in d.columns:
        d = d[d["model_name"].astype(str) == MODEL]
    else:
        # Infer model from path if possible.
        if MODEL not in path_s:
            continue

    if d.empty:
        continue

    d = d[["relation_key", dcol]].drop_duplicates().rename(columns={dcol: "candidate_delta"})
    m = fig2.merge(d, on="relation_key", how="inner")
    if len(m) < 10:
        continue

    try:
        r, rp = pearsonr(m["cos_improvement"], m["candidate_delta"])
        rho, sp = spearmanr(m["cos_improvement"], m["candidate_delta"])
        mad = (m["cos_improvement"] - m["candidate_delta"]).abs().mean()
    except Exception:
        continue

    rows.append({
        "n_overlap_fig2": len(m),
        "pearson_vs_fig2": r,
        "spearman_vs_fig2": rho,
        "mean_abs_diff": mad,
        "delta_col": dcol,
        "path": str(p),
        "dir": str(p.parent),
    })

res = pd.DataFrame(rows)
if res.empty:
    print("[WARN] no candidates found")
    raise SystemExit(0)

res = res.sort_values(
    ["n_overlap_fig2", "pearson_vs_fig2", "mean_abs_diff"],
    ascending=[False, False, True],
)

out = Path("runs/llama_fig2_step5_candidates_broad.csv")
res.to_csv(out, index=False)

print("[write]", out)
print()
print(res.head(30).to_string(index=False))
