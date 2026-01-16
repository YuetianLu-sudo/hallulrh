#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-model significance tests for the 4 Pearson scatter subfigures.

Input: a *summary* CSV with one row per (model_key, relation) containing at least:
  - model_key
  - relation
  - cos_improvement   (Δcos)
  - halluc_rate

Within each model (n = #relations), compute:
  - Pearson r
  - Parametric p-values for H0: r=0 using the standard t-test for correlation
      *two-sided* and *one-sided* (direction = 'greater' or 'less')
    (No Fisher-z CI; removed.)
  - Exact permutation p-values for Pearson r by permuting y
      *two-sided* and *one-sided* (same direction flag)
    *exact enumeration for n <= exact_max_n, else Monte Carlo*
  - Leave-one-relation-out (LOO) r range
  - Weighted Pearson r (weights = n_test, if present)

Outputs:
  - <out_dir>/scatter_panel_significance_by_model.csv
  - <out_dir>/scatter_panel_significance_latex.txt
"""
from __future__ import annotations

import argparse
import io
import itertools
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PRETTY_MODEL: Dict[str, str] = {
    "gemma_7b_it": "Gemma-7B-IT",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "Mistral-7B-Instruct",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
}


def read_csv_safely(path: str) -> pd.DataFrame:
    """Robust CSV reader (handles occasional odd bytes)."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def loo_range(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Leave-one-out range of Pearson r."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 4:
        return (float("nan"), float("nan"))
    rs: List[float] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rs.append(pearson_r(x[mask], y[mask]))
    return (float(np.nanmin(rs)), float(np.nanmax(rs)))


def weighted_pearson_r(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation with nonnegative weights."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.size != y.size or x.size != w.size or x.size < 2:
        return float("nan")
    w = np.clip(w, 0.0, None)
    if np.sum(w) <= 0:
        return float("nan")
    wx = np.sum(w * x) / np.sum(w)
    wy = np.sum(w * y) / np.sum(w)
    cov = np.sum(w * (x - wx) * (y - wy)) / np.sum(w)
    vx = np.sum(w * (x - wx) ** 2) / np.sum(w)
    vy = np.sum(w * (y - wy) ** 2) / np.sum(w)
    denom = math.sqrt(vx * vy)
    if denom == 0:
        return float("nan")
    return float(cov / denom)


# ---------------------------
# Parametric t-test for r=0
# ---------------------------

def _t_cdf(t: float, df: int) -> float:
    """
    Student-t CDF. Prefer SciPy; fall back to MPMath.
    """
    try:
        import scipy.stats as st  # type: ignore
        return float(st.t.cdf(t, df))
    except Exception:
        import mpmath as mp  # type: ignore

        t = float(t)
        df_f = float(df)
        x = df_f / (df_f + t * t)
        a = df_f / 2.0
        b = 0.5
        ib = mp.betainc(a, b, 0, x, regularized=True)
        if t >= 0:
            return float(1.0 - 0.5 * ib)
        return float(0.5 * ib)


def pearson_t_pvalues(r: float, n: int, direction: str) -> Tuple[float, float]:
    """
    Standard t-test p-values for H0: rho = 0 given observed Pearson r.

    Returns: (p_one_sided, p_two_sided)
      - one-sided alternative controlled by `direction`:
          'greater' : rho > 0
          'less'    : rho < 0
    """
    if n < 3 or not np.isfinite(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))

    df = n - 2
    t = r * math.sqrt(df / (1.0 - r * r))

    p_two = 2.0 * (1.0 - _t_cdf(abs(t), df))

    if direction == "greater":
        p_one = 1.0 - _t_cdf(t, df)
    elif direction == "less":
        p_one = _t_cdf(t, df)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    p_one = min(max(p_one, 0.0), 1.0)
    p_two = min(max(p_two, 0.0), 1.0)
    return (float(p_one), float(p_two))


# ---------------------------
# Permutation test (permute y)
# ---------------------------

def perm_pvalue_pearson(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str,
    exact_max_n: int,
    mc_samples: int,
    seed: int,
) -> Tuple[float, float, int, str]:
    """Permutation test p-value for Pearson r by permuting y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n != y.size or n < 3:
        return (float("nan"), float("nan"), 0, "na")

    r_obs = pearson_r(x, y)

    if n <= exact_max_n:
        rs = []
        for y_perm in itertools.permutations(y.tolist(), n):
            rs.append(pearson_r(x, np.asarray(y_perm, dtype=float)))
        rs = np.asarray(rs, dtype=float)
        n_perm = rs.size
        mode = "exact"
    else:
        rng = np.random.default_rng(seed)
        rs = np.empty(mc_samples, dtype=float)
        for i in range(mc_samples):
            rs[i] = pearson_r(x, rng.permutation(y))
        n_perm = mc_samples
        mode = "mc"

    if alternative == "two-sided":
        p = float(np.mean(np.abs(rs) >= abs(r_obs)))
    elif alternative == "greater":
        p = float(np.mean(rs >= r_obs))
    elif alternative == "less":
        p = float(np.mean(rs <= r_obs))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return (float(r_obs), float(p), int(n_perm), mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        default="analysis/chat_neutral_full_20251224_010241_post_v2/behavior_plus_lre.csv",
        help="CSV with one row per (model_key, relation).",
    )
    ap.add_argument("--out_dir", default="analysis/scatter_panel_significance")
    ap.add_argument("--x_col", default="cos_improvement")
    ap.add_argument("--y_col", default="halluc_rate")
    ap.add_argument("--group_col", default="model_key")
    ap.add_argument("--weight_col", default="n_test")
    ap.add_argument("--direction", choices=["greater", "less"], default="greater")
    ap.add_argument("--exact_max_n", type=int, default=9)
    ap.add_argument("--mc_samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"[ERROR] input_csv not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    df = read_csv_safely(args.input_csv)

    required = {args.group_col, "relation", args.x_col, args.y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df = df[np.isfinite(df[args.x_col]) & np.isfinite(df[args.y_col])]

    rows: List[Dict[str, object]] = []

    print("\n== Scatter-panel significance (within each model; across relations) ==")
    print(f"Input: {args.input_csv}")
    print(f"X={args.x_col}  Y={args.y_col}  group={args.group_col}")
    print(f"Directional one-sided alternative: {args.direction}")
    print()

    for g, sub in df.groupby(args.group_col):
        sub = sub.sort_values("relation")
        x = sub[args.x_col].to_numpy(dtype=float)
        y = sub[args.y_col].to_numpy(dtype=float)
        n = int(len(sub))

        r_obs = pearson_r(x, y)
        p_t_one, p_t_two = pearson_t_pvalues(r_obs, n, direction=args.direction)

        _, p_perm_two, n_perm, mode = perm_pvalue_pearson(
            x, y, alternative="two-sided",
            exact_max_n=args.exact_max_n, mc_samples=args.mc_samples, seed=args.seed
        )
        _, p_perm_one, _, _ = perm_pvalue_pearson(
            x, y, alternative=args.direction,
            exact_max_n=args.exact_max_n, mc_samples=args.mc_samples, seed=args.seed
        )

        loo_lo, loo_hi = loo_range(x, y)

        r_w = float("nan")
        if args.weight_col in sub.columns:
            w = sub[args.weight_col].to_numpy(dtype=float)
            r_w = weighted_pearson_r(x, y, w)

        pretty = PRETTY_MODEL.get(str(g), str(g))

        print(
            f"{pretty:22s}  n={n:d}  r={r_obs:+.3f}  "
            f"p_t(one)={p_t_one:.4f}  p_t(two)={p_t_two:.4f}  "
            f"p_perm(one)={p_perm_one:.4f}  p_perm(two)={p_perm_two:.4f}  "
            f"mode={mode} (n_perm={n_perm})"
        )

        rows.append(
            dict(
                model_key=str(g),
                model_pretty=pretty,
                n=n,
                r=r_obs,
                loo_r_min=loo_lo,
                loo_r_max=loo_hi,
                p_t_one=p_t_one,
                p_t_two=p_t_two,
                p_perm_one=p_perm_one,
                p_perm_two=p_perm_two,
                perm_mode=mode,
                n_perm=n_perm,
                r_weighted=r_w,
            )
        )

    out_df = pd.DataFrame(rows).sort_values("model_key")
    out_csv = os.path.join(args.out_dir, "scatter_panel_significance_by_model.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

    out_tex = os.path.join(args.out_dir, "scatter_panel_significance_latex.txt")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% LaTeX-ready snippet (one row per model)\n")
        f.write("% Model & r & p_t(two) & p_perm(two) \\\\\n")
        for _, rrow in out_df.iterrows():
            f.write(
                f"{rrow['model_pretty']} & "
                f"{float(rrow['r']):.3f} & "
                f"{float(rrow['p_t_two']):.4f} & "
                f"{float(rrow['p_perm_two']):.4f} \\\\\n"
            )
    print(f"Wrote: {out_tex}")

    print("\n== LaTeX snippet (copy/paste) ==")
    with open(out_tex, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
