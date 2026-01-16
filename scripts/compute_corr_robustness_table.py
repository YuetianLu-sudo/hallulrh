#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


MODEL_PRETTY = {
    "gemma_7b_it": "Gemma-7B-IT",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "Mistral-7B-Instruct",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
}


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom != 0 else float("nan")


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    # Average ranks for ties; then Pearson on ranks.
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson_r(rx, ry)


def fisher_ci(r: float, n: int, zcrit: float = 1.96) -> Tuple[float, float]:
    # Fisher z-transform CI for Pearson correlation.
    # For |r|==1, atanh is inf; handle safely.
    if not np.isfinite(r) or abs(r) >= 1.0:
        return (r, r)
    z = math.atanh(r)
    se = 1.0 / math.sqrt(max(n - 3, 1))
    lo = math.tanh(z - zcrit * se)
    hi = math.tanh(z + zcrit * se)
    return lo, hi


def loo_range(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    n = len(x)
    rs: List[float] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rs.append(pearson_r(x[mask], y[mask]))
    return float(np.min(rs)), float(np.max(rs))


def exact_permutation_p_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    # Enumerate all permutations of y (n!); two-sided p for Pearson r.
    robs = pearson_r(x, y)
    count = 0
    total = 0
    for perm in itertools.permutations(y):
        total += 1
        rperm = pearson_r(x, np.asarray(perm, dtype=float))
        if abs(rperm) >= abs(robs) - 1e-12:
            count += 1
    return count / total


def weighted_pearson_r(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    wsum = w.sum()
    if wsum <= 0:
        return float("nan")
    mx = (w * x).sum() / wsum
    my = (w * y).sum() / wsum
    cov = (w * (x - mx) * (y - my)).sum() / wsum
    vx = (w * (x - mx) ** 2).sum() / wsum
    vy = (w * (y - my) ** 2).sum() / wsum
    denom = math.sqrt(vx * vy)
    return float(cov / denom) if denom != 0 else float("nan")


def fishers_method(pvals: List[float]) -> float:
    # Combine p-values: stat = -2 sum log p ~ Chi^2_{2k}
    # Uses scipy if available; otherwise prints NaN.
    try:
        import scipy.stats as st
    except Exception:
        return float("nan")
    stat = -2.0 * sum(math.log(max(p, 1e-300)) for p in pvals)
    df = 2 * len(pvals)
    return float(st.chi2.sf(stat, df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to behavior_plus_lre.csv")
    ap.add_argument("--model_col", default="model_key")
    ap.add_argument("--relation_col", default="relation")
    ap.add_argument("--x_col", default="cos_improvement")
    ap.add_argument("--y_col", default="halluc_rate")
    ap.add_argument("--w_col", default="n_test")
    ap.add_argument("--emit", choices=["none", "latex", "both"], default="both")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Basic sanity checks
    for col in [args.model_col, args.relation_col, args.x_col, args.y_col, args.w_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.csv}")

    rows = []
    pvals = []

    for model_key, g in df.groupby(args.model_col):
        g = g.sort_values(args.relation_col)
        x = g[args.x_col].to_numpy(dtype=float)
        y = g[args.y_col].to_numpy(dtype=float)
        w = g[args.w_col].to_numpy(dtype=float)
        n = len(g)

        r = pearson_r(x, y)
        rho = spearman_rho(x, y)
        ci_lo, ci_hi = fisher_ci(r, n)
        loo_lo, loo_hi = loo_range(x, y)
        p_perm = exact_permutation_p_two_sided(x, y)
        r_w = weighted_pearson_r(x, y, w)

        pvals.append(p_perm)

        pretty = MODEL_PRETTY.get(model_key, str(model_key))
        rows.append({
            "model_key": model_key,
            "model": pretty,
            "n": n,
            "pearson_r": r,
            "spearman_rho": rho,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "loo_lo": loo_lo,
            "loo_hi": loo_hi,
            "p_perm": p_perm,
            "weighted_r": r_w,
        })

    out = pd.DataFrame(rows).sort_values("model")

    if args.emit in ("both", "latex"):
        print("== LaTeX rows for Table (corr robustness) ==")
        for _, r in out.iterrows():
            print(
                f"{r['model']} & "
                f"{r['pearson_r']:.3f} & {r['spearman_rho']:.3f} & "
                f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] & "
                f"[{r['loo_lo']:.3f},{r['loo_hi']:.3f}] & "
                f"{r['p_perm']:.3f} & {r['weighted_r']:.3f} \\\\"
            )

        p_fisher = fishers_method(pvals)
        if np.isfinite(p_fisher):
            print(f"\n== Fisher's method combined p over models ==\np = {p_fisher:.4f}")
        else:
            print("\n== Fisher's method combined p over models ==\n(scikit/scipy not available; skipped)")

    if args.emit in ("both", "none"):
        print("\n== Per-model stats ==")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
