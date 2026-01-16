#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid-search min_rel_denom thresholds for relation-level correlation analyses.

We assume you already have Step6 relation_summary.csv.gz (typically split=test),
with (per model, relation) columns including:
  - model_key, relation_key
  - n_total, n_value
  - refusal_rate
  - hall_rate_given_value
  - delta_cos_mean_all
  - delta_cos_mean_value

For each threshold T (1..max_denom), we compute for each model:
  Target A (hall_given_value):
    denom = n_value
    x = delta_cos_mean_value
    y = hall_rate_given_value
    r_stable, p_stable      : Pearson on relations with denom >= T
    r_stable_weighted, p_*  : Weighted Pearson on SAME filtered relations (weights=denom),
                              p via t-test with Kish effective n_eff.

  Target B (refusal):
    denom = n_total
    x = delta_cos_mean_all
    y = refusal_rate
    same r/p definitions as above.

Then per threshold we have 16 p-values:
  2 targets × 4 models × 2 r-types (stable + stable_weighted)

We compute mean_p16 = mean of these 16 p-values (missing -> 1.0),
rank thresholds by mean_p16, and output top10 plus detailed tables.

IMPORTANT: This is exploratory; do NOT pick threshold purely by min p for main paper result.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# ---- p-values via t distribution ----
try:
    from scipy import stats
except Exception as e:
    raise RuntimeError(
        "scipy is required for p-values. Please install: pip install -U scipy"
    ) from e


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """Unweighted Pearson r and two-sided p; require n>=3 else (nan,1,n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = int(x.size)
    if n < 3:
        return float("nan"), 1.0, n
    r, p = stats.pearsonr(x, y)
    # pearsonr may return nan if variance=0
    if not np.isfinite(r):
        return float("nan"), 1.0, n
    return float(r), float(p), n


def weighted_pearson_r(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float, float, int]:
    """
    Weighted Pearson correlation and approximate two-sided p-value.
    - weights: nonnegative
    - p-value uses t-test with df = n_eff - 2, where
        n_eff = (sum w)^2 / sum(w^2)   (Kish effective sample size)
    Returns: (r_w, p_w, n_eff, n_rel)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[m]; y = y[m]; w = w[m]
    n = int(x.size)
    if n < 3:
        return float("nan"), 1.0, float("nan"), n

    w_sum = float(np.sum(w))
    w2_sum = float(np.sum(w * w))
    n_eff = (w_sum * w_sum) / w2_sum if w2_sum > 0 else float("nan")

    # weighted means
    mx = float(np.sum(w * x) / w_sum)
    my = float(np.sum(w * y) / w_sum)

    # weighted cov/var
    cov = float(np.sum(w * (x - mx) * (y - my)) / w_sum)
    vx = float(np.sum(w * (x - mx) ** 2) / w_sum)
    vy = float(np.sum(w * (y - my) ** 2) / w_sum)

    if vx <= 0 or vy <= 0:
        return float("nan"), 1.0, n_eff, n

    r = cov / math.sqrt(vx * vy)

    # p-value approx with df = n_eff - 2
    if not np.isfinite(r):
        return float("nan"), 1.0, n_eff, n

    df = n_eff - 2.0
    if not np.isfinite(df) or df <= 1:
        return float(r), 1.0, n_eff, n

    # guard r=±1
    rr = max(min(float(r), 1.0 - 1e-12), -1.0 + 1e-12)
    t = rr * math.sqrt(df / (1.0 - rr * rr))
    p = 2.0 * stats.t.sf(abs(t), df)
    return float(r), float(p), float(n_eff), n


def run_one_target(df: pd.DataFrame, target: str, threshold: int) -> List[Dict]:
    """
    Returns rows for 4 models (or however many present).
    Each row includes r_stable/p_stable and r_stable_weighted/p_stable_weighted.
    """
    if target == "hall_given_value":
        denom_col = "n_value"
        x_col = "delta_cos_mean_value"
        y_col = "hall_rate_given_value"
    elif target == "refusal":
        denom_col = "n_total"
        x_col = "delta_cos_mean_all"
        y_col = "refusal_rate"
    else:
        raise ValueError(target)

    out_rows = []
    for mk in sorted(df["model_key"].unique()):
        g = df[df["model_key"] == mk].copy()

        # filter by denom
        g[denom_col] = pd.to_numeric(g[denom_col], errors="coerce")
        g = g[np.isfinite(g[denom_col])].copy()
        g = g[g[denom_col] >= threshold].copy()

        x = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
        w = pd.to_numeric(g[denom_col], errors="coerce").to_numpy(dtype=float)

        r_s, p_s, n_rel = pearson_r_p(x, y)
        r_w, p_w, n_eff, n_rel_w = weighted_pearson_r(x, y, w)

        out_rows.append({
            "threshold": threshold,
            "target": target,
            "model_key": mk,
            "denom_col": denom_col,
            "x_col": x_col,
            "y_col": y_col,
            "n_rel": n_rel,
            "r_stable": r_s,
            "p_stable": p_s,
            "r_stable_weighted": r_w,
            "p_stable_weighted": p_w,
            "n_eff_weighted": n_eff,
        })
    return out_rows


def mean_p16(rows_h: List[Dict], rows_r: List[Dict]) -> float:
    """
    Compute mean of 16 p-values:
      2 targets × 4 models × 2 r-types
    Missing/undefined -> 1.0
    """
    ps = []
    for rows in (rows_h, rows_r):
        for r in rows:
            for k in ("p_stable", "p_stable_weighted"):
                p = r.get(k, 1.0)
                if not np.isfinite(p):
                    p = 1.0
                ps.append(float(p))
    # enforce 16 length if fewer models present
    if len(ps) == 0:
        return 1.0
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation_summary", required=True,
                    help="Step6 relation_summary.csv.gz (split already applied in Step6 run)")
    ap.add_argument("--outdir", required=True, help="Where to write grid search CSVs")
    ap.add_argument("--min_threshold", type=int, default=1)
    ap.add_argument("--max_threshold", type=int, default=None,
                    help="Default: max over n_value and n_total")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.relation_summary)
    # Basic column sanity
    need = {
        "model_key", "relation_key",
        "n_total", "n_value",
        "refusal_rate", "hall_rate_given_value",
        "delta_cos_mean_all", "delta_cos_mean_value",
    }
    missing = [c for c in sorted(need) if c not in df.columns]
    if missing:
        raise ValueError(f"relation_summary missing columns: {missing}\nFound: {list(df.columns)}")

    # define thresholds
    max_d = int(
        np.nanmax([
            pd.to_numeric(df["n_total"], errors="coerce").max(),
            pd.to_numeric(df["n_value"], errors="coerce").max(),
        ])
    )
    min_t = max(1, int(args.min_threshold))
    max_t = int(args.max_threshold) if args.max_threshold is not None else max_d
    max_t = max(min_t, max_t)

    all_rows = []
    thr_rows = []

    for T in range(min_t, max_t + 1):
        rows_h = run_one_target(df, "hall_given_value", T)
        rows_r = run_one_target(df, "refusal", T)

        mp = mean_p16(rows_h, rows_r)
        thr_rows.append({"threshold": T, "mean_p16": mp})
        all_rows.extend(rows_h)
        all_rows.extend(rows_r)

    df_thr = pd.DataFrame(thr_rows).sort_values("mean_p16", ascending=True).reset_index(drop=True)
    df_all = pd.DataFrame(all_rows)

    # Write full grid
    full_thr_path = outdir / "gridsearch_all_thresholds.csv"
    full_tbl_path = outdir / "gridsearch_all_tables.csv"
    df_thr.to_csv(full_thr_path, index=False)
    df_all.to_csv(full_tbl_path, index=False)
    print(f"[ok] wrote: {full_thr_path}")
    print(f"[ok] wrote: {full_tbl_path}")

    # TopK thresholds
    topk = int(args.topk)
    df_top = df_thr.head(topk).copy()
    top_thr = set(df_top["threshold"].tolist())
    df_top_tbl = df_all[df_all["threshold"].isin(top_thr)].copy()

    top_thr_path = outdir / f"gridsearch_top{topk}_thresholds.csv"
    top_tbl_path = outdir / f"gridsearch_top{topk}_tables.csv"
    df_top.to_csv(top_thr_path, index=False)
    df_top_tbl.to_csv(top_tbl_path, index=False)

    print(f"[ok] wrote: {top_thr_path}")
    print(f"[ok] wrote: {top_tbl_path}")

    # Pretty print topK
    print("\n== Top thresholds by mean_p16 (smaller => more significant on average) ==")
    for i, r in df_top.iterrows():
        print(f"rank {i+1:2d}: threshold={int(r['threshold'])}  mean_p16={float(r['mean_p16']):.6g}")

    print("\nNote: 'stable_weighted' is computed AFTER threshold filtering (weights=denom), "
          "with p-value via Kish n_eff approximation.\n")


if __name__ == "__main__":
    main()
