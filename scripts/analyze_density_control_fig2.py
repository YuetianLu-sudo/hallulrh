#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute correlations + partial correlations to test whether Fig2's Δcos↔Hall|Value
association survives after controlling for target density.

Input: points_deltacos_vs_density_*.csv (produced by plot_deltacos_vs_target_density.py)
Required cols:
  - model_key, relation_key
  - cos_improvement
  - target_density
  - ycol (default: hall_rate_answered)

Outputs:
  - density_control_<ycol>.csv in --outdir (default: same dir as points file)
"""

import argparse
import glob
import math
import os
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, t  # type: ignore
except Exception:
    pearsonr = None
    t = None


def _autofind_points() -> str:
    pats = [
        "runs/experiments/**/points_deltacos_vs_density_fig2_all47_min11_intersection.csv",
        "runs/experiments/**/points_deltacos_vs_density*min11*intersection*.csv",
        "runs/experiments/**/points_deltacos_vs_density*.csv",
    ]
    cands = []
    for pat in pats:
        cands.extend(glob.glob(pat, recursive=True))
    cands = [p for p in cands if os.path.isfile(p)]
    if not cands:
        raise SystemExit("[error] cannot find any points_deltacos_vs_density*.csv under runs/experiments/")
    # pick newest by mtime
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def _corr_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 3:
        return float("nan"), float("nan"), n
    if pearsonr is None:
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float("nan"), n
    r, p = pearsonr(x, y)  # two-sided
    return float(r), float(p), n


def _partial_corr_r_p(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, int]:
    """
    Partial corr(x,y|z): correlate residuals after regressing x~z and y~z.
    p-value via t-test with df = n-3.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[m]
    y = y[m]
    z = z[m]
    n = int(x.size)
    if n < 4:
        return float("nan"), float("nan"), n

    Z = np.column_stack([np.ones(n), z])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z.dot(bx)
    ry = y - Z.dot(by)

    r, p, _ = _corr_r_p(rx, ry)
    if not np.isfinite(r) or t is None:
        return r, float("nan"), n

    dfree = n - 3
    # guard numerical
    denom = max(1e-12, 1.0 - r * r)
    tstat = r * math.sqrt(dfree / denom)
    p2 = 2.0 * (1.0 - t.cdf(abs(tstat), dfree))
    return float(r), float(p2), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=None, help="points_deltacos_vs_density*.csv (auto-find if omitted)")
    ap.add_argument("--outdir", default=None, help="output dir (default: same dir as points file)")
    ap.add_argument("--ycol", default="hall_rate_answered",
                    help="Outcome column. e.g. hall_rate_answered (Hall|Value) or hall_rate_noncorrect (Hall|Hall+Ref)")
    args = ap.parse_args()

    points_path = args.points or _autofind_points()
    df = pd.read_csv(points_path)

    need = {"model_key", "relation_key", "cos_improvement", "target_density", args.ycol}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[error] missing columns in points CSV: {sorted(miss)}")

    outdir = args.outdir or os.path.dirname(points_path)
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, f"density_control_{args.ycol}.csv")

    rows = []

    # per-model
    for mk, sub in df.groupby("model_key"):
        r_xy, p_xy, _ = _corr_r_p(sub["cos_improvement"], sub[args.ycol])
        r_xz, p_xz, _ = _corr_r_p(sub["cos_improvement"], sub["target_density"])
        r_yz, p_yz, _ = _corr_r_p(sub[args.ycol], sub["target_density"])
        pr, pp, _ = _partial_corr_r_p(sub["cos_improvement"], sub[args.ycol], sub["target_density"])

        rows.append({
            "model_key": str(mk),
            "n_rel": int(sub["relation_key"].nunique()),
            "r_deltacos_vs_y": r_xy,
            "p_deltacos_vs_y": p_xy,
            "r_deltacos_vs_density": r_xz,
            "p_deltacos_vs_density": p_xz,
            "r_density_vs_y": r_yz,
            "p_density_vs_y": p_yz,
            "partial_r_deltacos_vs_y_given_density": pr,
            "p_partial": pp,
        })

    # pooled FE (demean within model)
    g = df["model_key"].astype(str)
    x_res = df["cos_improvement"] - df.groupby(g)["cos_improvement"].transform("mean")
    y_res = df[args.ycol] - df.groupby(g)[args.ycol].transform("mean")
    z_res = df["target_density"] - df.groupby(g)["target_density"].transform("mean")
    pr_fe, pp_fe, n_fe = _partial_corr_r_p(x_res.to_numpy(float), y_res.to_numpy(float), z_res.to_numpy(float))

    rows.append({
        "model_key": "POOLED_FE",
        "n_rel": int(df["relation_key"].nunique()),
        "r_deltacos_vs_y": float("nan"),
        "p_deltacos_vs_y": float("nan"),
        "r_deltacos_vs_density": float("nan"),
        "p_deltacos_vs_density": float("nan"),
        "r_density_vs_y": float("nan"),
        "p_density_vs_y": float("nan"),
        "partial_r_deltacos_vs_y_given_density": pr_fe,
        "p_partial": pp_fe,
        "n_points": int(n_fe),
    })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print("[done] points:", points_path)
    print("[done] wrote:", out_csv)
    print(out.to_string(index=False))

    if pearsonr is None:
        print("[WARN] SciPy not found: p-values for raw correlations are NaN.")
    if t is None:
        print("[WARN] SciPy not found: p-values for partial correlations are NaN.")


if __name__ == "__main__":
    main()
