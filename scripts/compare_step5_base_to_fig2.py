import gzip
import os
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

variant = os.environ.get("RUN_VARIANT", "plain")
latest_path = Path(f"runs/step5_all47_translation_{variant}.latest")

if not latest_path.exists():
    raise FileNotFoundError(latest_path)

base_root = Path(latest_path.read_text(encoding="utf-8").strip())

fig2_path = Path(
    "runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209/analysis_all47/lre3way_behavior_plus_deltacos_all47.csv"
)
fig2 = pd.read_csv(fig2_path)

if "model_key" not in fig2.columns and "model_name" in fig2.columns:
    fig2 = fig2.rename(columns={"model_name": "model_key"})
if "relation_key" not in fig2.columns and "relation" in fig2.columns:
    fig2 = fig2.rename(columns={"relation": "relation_key"})

fig2 = fig2[fig2["n_test"] > 10].copy()
fig2 = fig2[["model_key", "relation_key", "cos_improvement"]]

models = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

rows = []
env_lines = []

for mk in models:
    p = base_root / mk / "relation_summary.csv.gz"
    if not p.exists():
        raise FileNotFoundError(p)

    with gzip.open(p, "rt", encoding="utf-8") as f:
        b = pd.read_csv(f)

    if "cos_improvement" in b.columns:
        dcol = "cos_improvement"
    elif "delta_cos_mean_test" in b.columns:
        dcol = "delta_cos_mean_test"
    elif "delta_cos" in b.columns:
        dcol = "delta_cos"
    else:
        raise KeyError(f"No delta/cos column in {p}: {list(b.columns)}")

    b = b[["relation_key", dcol]].rename(columns={dcol: "base_delta"})
    f2 = fig2[fig2["model_key"] == mk]
    m = f2.merge(b, on="relation_key", how="inner")

    r, rp = pearsonr(m["cos_improvement"], m["base_delta"])
    rho, sp = spearmanr(m["cos_improvement"], m["base_delta"])
    mad = (m["cos_improvement"] - m["base_delta"]).abs().mean()

    rows.append(
        {
            "model_key": mk,
            "n_overlap_fig2": len(m),
            "pearson_vs_fig2": r,
            "spearman_vs_fig2": rho,
            "mean_abs_diff": mad,
            "base_dir": str(base_root / mk),
        }
    )
    env_lines.append(f"BASE_{mk}='{base_root / mk}'")

res = pd.DataFrame(rows)
out = Path(f"runs/step5_all47_translation_{variant}_vs_fig2.csv")
res.to_csv(out, index=False)

print("[write]", out)
print(res.to_string(index=False))

ok = (res["n_overlap_fig2"].eq(28) & (res["pearson_vs_fig2"] > 0.995)).all()

if ok:
    Path("runs/fig2_step5_base_paths.env").write_text(
        "\n".join(env_lines) + "\n", encoding="utf-8"
    )
    print("[OK] This base matches Figure 2 sufficiently. Wrote runs/fig2_step5_base_paths.env")
else:
    print("[WARN] This base does not fully match Figure 2. Do not run affine from it yet.")
