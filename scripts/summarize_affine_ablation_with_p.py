#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr  # type: ignore
except Exception:
    pearsonr = None
    spearmanr = None


MODEL_ORDER = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]


def _corr_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    if pearsonr is None:
        return float(np.corrcoef(x, y)[0, 1]), float("nan")
    r, p = pearsonr(x, y)  # two-sided
    return float(r), float(p)


def _corr_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    if spearmanr is None:
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)  # two-sided (SciPy)
    return float(rho), float(p)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="Path to fig2_points_with_affine.csv")
    ap.add_argument("--ycol", default="hall_rate_answered",
                    help="Outcome column. Default hall_rate_answered (Hall/(Hall+Correct)).")
    ap.add_argument("--out", default=None, help="Output CSV path (default: alongside points).")
    args = ap.parse_args()

    df = pd.read_csv(args.points)

    need = {"model_key", "relation_key", "cos_improvement", "cos_improvement_affine", args.ycol}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[error] missing columns: {sorted(miss)}. have={list(df.columns)}")

    rows = []
    for mk in MODEL_ORDER:
        sub = df[df["model_key"].astype(str) == mk].copy()
        if sub.empty:
            continue

        A = sub["cos_improvement"].to_numpy(float)
        B = sub["cos_improvement_affine"].to_numpy(float)
        Y = sub[args.ycol].to_numpy(float)

        r_ab, p_ab = _corr_pearson(A, B)
        rho_ab, p_rho_ab = _corr_spearman(A, B)

        r_a, p_a = _corr_pearson(A, Y)
        r_b, p_b = _corr_pearson(B, Y)

        rows.append({
            "model_key": mk,
            "n_rel": int(len(sub)),

            "pearson_A_vs_B": r_ab,
            "p_pearson_A_vs_B": p_ab,
            "spearman_A_vs_B": rho_ab,
            "p_spearman_A_vs_B": p_rho_ab,

            f"pearson_A_vs_{args.ycol}": r_a,
            f"p_pearson_A_vs_{args.ycol}": p_a,
            f"pearson_B_vs_{args.ycol}": r_b,
            f"p_pearson_B_vs_{args.ycol}": p_b,
        })

    out_df = pd.DataFrame(rows)
    if args.out is None:
        out_path = os.path.join(os.path.dirname(args.points), f"affine_ablation_summary_with_p_{args.ycol}.csv")
    else:
        out_path = args.out

    out_df.to_csv(out_path, index=False)
    print("[done] wrote:", out_path)

    # pretty print
    print("\n== affine ablation summary (with p-values) ==")
    for _, r in out_df.iterrows():
        mk = r["model_key"]
        print(f"\n{mk} (n_rel={int(r['n_rel'])})")
        print(f"  A vs B:  Pearson r={r['pearson_A_vs_B']:.6f} (p={fmt_p(r['p_pearson_A_vs_B'])}), "
              f"Spearman ρ={r['spearman_A_vs_B']:.6f} (p={fmt_p(r['p_spearman_A_vs_B'])})")
        print(f"  A vs Y:  Pearson r={r[f'pearson_A_vs_{args.ycol}']:.6f} (p={fmt_p(r[f'p_pearson_A_vs_{args.ycol}'])})")
        print(f"  B vs Y:  Pearson r={r[f'pearson_B_vs_{args.ycol}']:.6f} (p={fmt_p(r[f'p_pearson_B_vs_{args.ycol}'])})")

    if pearsonr is None or spearmanr is None:
        print("\n[WARN] SciPy not found -> some p-values may be NaN. Install via: pip install scipy")


if __name__ == "__main__":
    main()
