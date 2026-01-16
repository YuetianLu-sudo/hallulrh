#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute pooled partial correlations for the gold-answer sanity check table.

We compute partial correlations between delta_cos and:
  - value_rate
  - wrong_rate
  - acc_given_value
after residualizing out model fixed effects (model dummies + intercept).

This reproduces the numbers used in Appendix B text, e.g.:
  r=0.411 (value rate), r=-0.561 (wrong rate), r=0.606 (Acc|Value)
and the variant excluding country_language.

Example:
  python scripts/compute_gold_sanity_check_pooled_partial.py \
    --by_relation analysis/.../gold_sanity_check_by_relation.csv \
    --drop_relation country_language
"""

import argparse
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _try_p_value_partial_r(r: float, n: int, k: int) -> Optional[float]:
    """
    Two-sided p-value for partial correlation with k controls:
      t = r * sqrt((n-k-2) / (1-r^2)), df = n-k-2
    Returns None if scipy is not available or df <= 0.
    """
    df = n - k - 2
    if df <= 0 or not np.isfinite(r) or abs(r) >= 1.0:
        return None
    try:
        from scipy import stats  # type: ignore
        t = r * math.sqrt(df / max(1e-12, (1.0 - r * r)))
        p = 2.0 * stats.t.sf(abs(t), df=df)
        return float(p)
    except Exception:
        return None


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals y - X beta."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr_with_model_fe(df: pd.DataFrame, y_col: str, x_col: str = "delta_cos") -> Tuple[float, int, int]:
    """
    Partial correlation corr(resid(x), resid(y)) controlling for model fixed effects.
    Controls: model dummies (k = #models - 1). Intercept is always included.
    Returns: (r, n, k_controls)
    """
    df = df.dropna(subset=[x_col, y_col, "model_key"]).copy()
    n = len(df)
    models = sorted(df["model_key"].unique().tolist())
    if len(models) < 2:
        return float("nan"), n, 0

    # Design matrix: intercept + (M-1) dummies
    X = np.ones((n, 1), dtype=np.float64)
    for m in models[1:]:
        X = np.column_stack([X, (df["model_key"].values == m).astype(np.float64)])

    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)

    rx = residualize(x, X)
    ry = residualize(y, X)

    # Pearson r on residuals
    r = float(np.corrcoef(rx, ry)[0, 1])
    k_controls = len(models) - 1
    return r, n, k_controls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by_relation", required=True, help="Path to gold_sanity_check_by_relation.csv")
    ap.add_argument("--drop_relation", default="", help="Optional relation name to drop, e.g., country_language")
    args = ap.parse_args()

    df = pd.read_csv(args.by_relation)

    required = {"model_key", "relation", "delta_cos", "value_rate", "wrong_rate", "acc_given_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")

    print("== Input ==")
    print(f"CSV: {args.by_relation}")
    print(f"Rows: {len(df)}  Models: {df['model_key'].nunique()}  Relations: {df['relation'].nunique()}")

    # Full pooled
    print("\n== Pooled partial correlations (controls: model fixed effects) ==")
    for y_col, label in [
        ("value_rate", "value rate"),
        ("wrong_rate", "wrong rate"),
        ("acc_given_value", "Acc|Value"),
    ]:
        r, n, k = partial_corr_with_model_fe(df, y_col=y_col)
        p = _try_p_value_partial_r(r, n=n, k=k)
        if p is None:
            print(f"Δcos ~ {label:9s}: r={r:.3f} (n={n}, k={k})")
        else:
            print(f"Δcos ~ {label:9s}: r={r:.3f} (p={p:.4g}; n={n}, k={k})")

    # Drop one relation (optional)
    drop = args.drop_relation.strip()
    if drop:
        df2 = df[df["relation"] != drop].copy()
        print(f"\n== Excluding relation: {drop} ==")
        for y_col, label in [
            ("value_rate", "value rate"),
            ("wrong_rate", "wrong rate"),
            ("acc_given_value", "Acc|Value"),
        ]:
            r, n, k = partial_corr_with_model_fe(df2, y_col=y_col)
            p = _try_p_value_partial_r(r, n=n, k=k)
            if p is None:
                print(f"Δcos ~ {label:9s}: r={r:.3f} (n={n}, k={k})")
            else:
                print(f"Δcos ~ {label:9s}: r={r:.3f} (p={p:.4g}; n={n}, k={k})")

    # LaTeX-ready snippet for the paragraph
    r_val, _, _ = partial_corr_with_model_fe(df, "value_rate")
    r_wrong, _, _ = partial_corr_with_model_fe(df, "wrong_rate")
    r_acc, _, _ = partial_corr_with_model_fe(df, "acc_given_value")

    print("\n== LaTeX snippet (pooled partial r; model FE) ==")
    print(
        "Pooling all 24 model$\\times$relation points and residualizing out model fixed effects "
        f"yields partial correlations of $r={r_val:.3f}$ (value rate), "
        f"$r={r_wrong:.3f}$ (wrong rate), and $r={r_acc:.3f}$ ($\\mathrm{{Acc}}\\mid\\mathrm{{Value}}$)."
    )

    if drop:
        df2 = df[df["relation"] != drop].copy()
        r_wrong2, _, _ = partial_corr_with_model_fe(df2, "wrong_rate")
        r_acc2, _, _ = partial_corr_with_model_fe(df2, "acc_given_value")
        print(
            f"Excluding the smallest LRE relation (\\texttt{{{drop}}}) preserves the accuracy effects "
            f"($r={r_wrong2:.3f}$ for wrong rate; $r={r_acc2:.3f}$ for $\\mathrm{{Acc}}\\mid\\mathrm{{Value}}$)."
        )


if __name__ == "__main__":
    main()
