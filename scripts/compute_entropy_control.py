#!/usr/bin/env python3
"""
Compute output-space concentration proxies (Top-1 share, normalized entropy) from
*with_judge*.csv files, and compute within-model partial correlations between
Δcos and hallucination rate controlling for these proxies.

This reproduces Appendix~{entropy_control}-style tables.

Inputs:
1) behavior_plus_lre.csv (one row per model×relation) containing at least:
   - model_name
   - relation (or task)
   - cos_improvement (Δcos)  [or delta_cos]
   - hallucination_rate      [or hall_rate]
2) with_judge CSVs produced by the judge pipeline (glob), containing:
   - task
   - model_name  (may include suffixes _baseline/_relpanel)
   - answer
   - judge_label  ("HALLUCINATION" / "REFUSAL")

Proxy definitions (as in the paper text):
- Top1_{m,r} = max_a count(a) / |A_{m,r}| over hallucinated answers
- Entropy_{m,r} = normalized Shannon entropy over answers in A_{m,r}
    H = -Σ p log p
    H_norm = H / log(K) where K = #unique answers (if K<=1 -> 0)

Partial correlation (within each model, n=6 relations):
- Compute residuals of Δcos and hallucination_rate after regressing out the proxy
  (with intercept), then correlate residuals.

Also prints pooled partial correlation over all 24 points by residualizing out
model fixed effects + the proxy (matching the appendix paragraph).

Usage:
  python scripts/compute_entropy_control.py \
    --behavior_plus_lre analysis/.../behavior_plus_lre.csv \
    --glob "data/judge_inputs/**/*with_judge*.csv"

"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns exist: {list(candidates)}. Available: {list(df.columns)}")
    return None


def _canonical_base_model(name: str) -> str:
    for suf in ("_baseline", "_relpanel"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _normalize_answer(s: str) -> str:
    # Minimal normalization to reduce pure whitespace noise, while keeping the
    # metric faithful to "exact surface form" concentration.
    return " ".join(str(s).strip().split())


def _top1_and_entropy_norm(answers: List[str]) -> Tuple[float, float]:
    n = len(answers)
    if n == 0:
        return 0.0, 0.0
    # counts
    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    top1 = max(counts.values()) / n
    K = len(counts)
    if K <= 1:
        return float(top1), 0.0
    ps = np.array(list(counts.values()), dtype=float) / n
    H = float(-(ps * np.log(ps)).sum())
    H_norm = H / math.log(K)
    return float(top1), float(H_norm)


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) == 0:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y after OLS fit on X (includes intercept if X has it)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _partial_corr_resid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    # add intercept
    X = np.column_stack([np.ones(len(z)), z])
    rx = _residualize(x, X)
    ry = _residualize(y, X)
    return _pearsonr(rx, ry)


def _pooled_partial_corr_with_fixed_effects(
    df: pd.DataFrame,
    *,
    col_model: str,
    col_x: str,
    col_y: str,
    col_z: str,
) -> float:
    """
    Pooled partial correlation across all points controlling for:
      - model fixed effects (dummies)
      - proxy z
    Using residualization: regress x and y on [intercept, model dummies, z],
    then correlate residuals.
    """
    models = sorted(df[col_model].unique().tolist())
    # drop one dummy to avoid singular matrix (reference category)
    ref = models[0]
    dummies = []
    for m in models[1:]:
        dummies.append((df[col_model] == m).astype(float).to_numpy())
    Z = df[col_z].to_numpy(dtype=float)
    X_design = [np.ones(len(df))]
    X_design.extend(dummies)
    X_design.append(Z)
    X = np.column_stack(X_design)

    x = df[col_x].to_numpy(dtype=float)
    y = df[col_y].to_numpy(dtype=float)
    rx = _residualize(x, X)
    ry = _residualize(y, X)
    return _pearsonr(rx, ry)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior_plus_lre", required=True, help="CSV with Δcos and hallucination rates per model×relation.")
    ap.add_argument("--glob", dest="glob_pat", required=True, help="Glob for *with_judge*.csv files.")
    ap.add_argument("--ndigits", type=int, default=3, help="Decimals for printed correlations.")
    args = ap.parse_args()

    beh = pd.read_csv(args.behavior_plus_lre)

    col_model_b = _pick_col(beh, ["model_name", "model", "model_key"])
    col_rel_b = _pick_col(beh, ["relation", "task"])
    col_x = _pick_col(beh, ["cos_improvement", "delta_cos", "deltacos", "DeltaCos"])
    col_y = _pick_col(beh, ["hallucination_rate", "hall_rate", "p_hall", "halluc_rate"])

    beh[col_model_b] = beh[col_model_b].astype(str).map(_canonical_base_model)
    beh[col_rel_b] = beh[col_rel_b].astype(str)

    # Load judged outputs
    paths = sorted(glob.glob(args.glob_pat, recursive=True))
    if not paths:
        print(f"[ERROR] No files matched: {args.glob_pat}", file=sys.stderr)
        return 2

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}", file=sys.stderr)
            continue
        dfs.append(df)
    if not dfs:
        print("[ERROR] Could not read any with_judge CSVs.", file=sys.stderr)
        return 2
    out = pd.concat(dfs, ignore_index=True)

    col_model_o = _pick_col(out, ["model_name", "model", "model_key"])
    col_rel_o = _pick_col(out, ["task", "relation"])
    col_ans = _pick_col(out, ["answer"])
    col_label = _pick_col(out, ["judge_label", "label"])

    out[col_model_o] = out[col_model_o].astype(str).map(_canonical_base_model)
    out[col_rel_o] = out[col_rel_o].astype(str)

    # Compute proxies per (model, relation)
    proxies = []
    for (m, r), g in out.groupby([col_model_o, col_rel_o], sort=False):
        g_h = g[g[col_label].astype(str).str.upper() == "HALLUCINATION"]
        answers = [_normalize_answer(a) for a in g_h[col_ans].tolist()]
        top1, ent = _top1_and_entropy_norm(answers)
        proxies.append((m, r, top1, ent, len(answers)))
    prox = pd.DataFrame(proxies, columns=["model_name", "relation", "top1", "entropy_norm", "n_hall"])

    # Merge with behavior table
    merged = beh.merge(
        prox,
        how="left",
        left_on=[col_model_b, col_rel_b],
        right_on=["model_name", "relation"],
        suffixes=("", "_prox"),
    )

    # Fill missing proxies (e.g., if no hallucinations) with 0
    merged["top1"] = merged["top1"].fillna(0.0)
    merged["entropy_norm"] = merged["entropy_norm"].fillna(0.0)

    # Within-model correlations
    rows = []
    for m, g in merged.groupby(col_model_b, sort=False):
        x = g[col_x].to_numpy(dtype=float)
        y = g[col_y].to_numpy(dtype=float)
        r_xy = _pearsonr(x, y)
        r_top1 = _partial_corr_resid(x, y, g["top1"].to_numpy(dtype=float))
        r_ent = _partial_corr_resid(x, y, g["entropy_norm"].to_numpy(dtype=float))
        rows.append((m, r_xy, r_top1, r_ent))

    # Print summary + LaTeX snippet
    def fmt(v: float) -> str:
        if math.isnan(v):
            return "nan"
        return f"{v:.{args.ndigits}f}"

    print("== Within-model partial correlations ==")
    for m, r_xy, r_top1, r_ent in rows:
        print(f"{m}: Pearson r={fmt(r_xy)}  Partial r (Top1)={fmt(r_top1)}  Partial r (Entropy)={fmt(r_ent)}")

    pooled_top1 = _pooled_partial_corr_with_fixed_effects(
        merged, col_model=col_model_b, col_x=col_x, col_y=col_y, col_z="top1"
    )
    pooled_ent = _pooled_partial_corr_with_fixed_effects(
        merged, col_model=col_model_b, col_x=col_x, col_y=col_y, col_z="entropy_norm"
    )
    print("\n== Pooled partial correlation (controls: model fixed effects + proxy) ==")
    print(f"Pooled partial r (Top1 control)   = {fmt(pooled_top1)}")
    print(f"Pooled partial r (Entropy control)= {fmt(pooled_ent)}")

    print("\n== LaTeX snippet for Table rows ==")
    for m, r_xy, r_top1, r_ent in rows:
        # Pretty model names for the paper could be mapped outside the script.
        print(f"{m} & {fmt(r_xy)} & {fmt(r_top1)} & {fmt(r_ent)} \\\\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
