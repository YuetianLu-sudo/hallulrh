#!/usr/bin/env python3
"""
Make Figure-2-style scatter panels for NATURAL LRE using 3-way judge labels.

Goal:
  Replicate the paper's Figure 2 style:
    x = Δcos (cos_improvement)
    y = hallucination rate among VALUE answers: hall / (hall + correct)
  using your 3-way judge outputs (REFUSAL/CORRECT/HALLUCINATION).

Input CSV:
  Typically: lre3way_behavior_plus_deltacos_all47.csv (188 rows = 47*4)
  Expected flexible columns:
    - model_key or model_name
    - relation_key or relation or task
    - cos_improvement (Δcos)
    - n_test (for filtering, from deltacos)
    - counts or rate columns:
        * hall_rate_given_value (preferred if present)
        * else counts: hall/hallucination + correct + refusal

Outputs:
  - scatter_panel_4models_hall_rate_given_value_fit.(pdf|png)
  - correlation_summary_hall_rate_given_value.csv
"""

import argparse
import os
from typing import Dict, Optional, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

MODEL_TITLE = {
    "gemma_7b_it": "Gemma-7B-IT",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "Mistral-7B-Instruct",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
}


def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {}
    if "model_name" in df.columns and "model_key" not in df.columns:
        ren["model_name"] = "model_key"
    if "relation" in df.columns and "relation_key" not in df.columns:
        ren["relation"] = "relation_key"
    if "task" in df.columns and "relation_key" not in df.columns:
        ren["task"] = "relation_key"
    if "delta_cos_mean_test" in df.columns and "cos_improvement" not in df.columns:
        ren["delta_cos_mean_test"] = "cos_improvement"
    if ren:
        df = df.rename(columns=ren)
    return df


def _first_existing(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _ensure_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have:
      - hall_rate_given_value = hall/(hall+correct)
    Either use an existing column or compute from counts.
    """
    df = df.copy()

    # If already present, just trust it.
    if "hall_rate_given_value" in df.columns:
        df["hall_rate_given_value"] = pd.to_numeric(df["hall_rate_given_value"], errors="coerce")
        return df

    # Otherwise compute from counts.
    hall_col = _first_existing(df, ["n_hallucination", "hallucination", "hall", "n_hall"])
    cor_col  = _first_existing(df, ["n_correct", "correct", "n_cor"])
    if hall_col is None or cor_col is None:
        raise ValueError(
            "Cannot compute hall_rate_given_value: missing hall/correct counts. "
            f"Found cols={list(df.columns)}"
        )

    hall = pd.to_numeric(df[hall_col], errors="coerce")
    cor  = pd.to_numeric(df[cor_col], errors="coerce")
    denom = hall + cor
    df["hall_rate_given_value"] = np.where(denom > 0, hall / denom, np.nan)
    return df


def pearson_r_p_two(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        return float("nan"), float("nan")
    try:
        import scipy.stats as st  # type: ignore
        r, p = st.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float("nan")


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return float("nan"), float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def save_figure(fig: plt.Figure, base_path_no_ext: str, dpi: int = 300) -> None:
    fig.savefig(base_path_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path_no_ext + ".pdf", bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (e.g., lre3way_behavior_plus_deltacos_all47.csv)")
    ap.add_argument("--outdir", required=True, help="Output directory for plots + summary")
    ap.add_argument("--min-n-test", type=int, default=11,
                    help="Filter relations with n_test >= this. Use 11 to match 'remove 10 or fewer'.")
    ap.add_argument("--intersection", action="store_true",
                    help="Use intersection of relations across the 4 models (recommended for Fig.2 comparability).")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = _rename_cols(df)

    need = {"model_key", "relation_key", "cos_improvement"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}. Found: {list(df.columns)}")

    # numeric
    df["cos_improvement"] = pd.to_numeric(df["cos_improvement"], errors="coerce")

    # ensure y metric
    df = _ensure_rate_columns(df)

    # n_test filter (if available)
    if "n_test" in df.columns:
        df["n_test"] = pd.to_numeric(df["n_test"], errors="coerce")
        df = df[df["n_test"] >= int(args.min_n_test)].copy()
    else:
        print("[warn] n_test column not found; skipping n_test filtering")

    # keep only our 4 models
    df = df[df["model_key"].isin(MODEL_ORDER)].copy()

    # drop rows with missing x/y
    df = df[np.isfinite(df["cos_improvement"].to_numpy(dtype=float))].copy()
    df = df[np.isfinite(df["hall_rate_given_value"].to_numpy(dtype=float))].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering. Check inputs/filters.")

    # intersection across models (recommended for Fig.2)
    if args.intersection:
        sets: List[Set[str]] = []
        for mk in MODEL_ORDER:
            s = set(df.loc[df["model_key"] == mk, "relation_key"].astype(str).tolist())
            sets.append(s)
        inter = set.intersection(*sets) if sets else set()
        df = df[df["relation_key"].astype(str).isin(inter)].copy()
        print(f"[filter] intersection enabled: |I|={len(inter)} relations")
    else:
        print("[filter] intersection disabled")

    # report n_rel per model
    print("[info] #relations per model (after filters):")
    for mk in MODEL_ORDER:
        n_rel = df[df["model_key"] == mk]["relation_key"].nunique()
        print(f"  {mk}: {n_rel}")

    # global x range
    gxmin = float(df["cos_improvement"].min()) - 0.03
    gxmax = float(df["cos_improvement"].max()) + 0.03
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    corr_rows = []

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub["cos_improvement"].to_numpy(dtype=float)
        y = sub["hall_rate_given_value"].to_numpy(dtype=float)

        r, p = pearson_r_p_two(x, y)
        slope, intercept = fit_line(x, y)

        ax.scatter(
            x, y,
            s=55,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(gxmin, gxmax, 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, linestyle="--", linewidth=1.6, color="black", alpha=0.55, zorder=2)

        title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f}"
        if np.isfinite(p):
            title += f", p={p:.4f})"
        else:
            title += ")"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(gxmin, gxmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

        corr_rows.append({
            "model_key": mk,
            "n_rel": int(sub["relation_key"].nunique()),
            "pearson_r": float(r),
            "p_two_sided": float(p) if np.isfinite(p) else np.nan,
            "slope": float(slope),
            "intercept": float(intercept),
            "min_n_test": int(args.min_n_test) if "n_test" in sub.columns else np.nan,
            "intersection": bool(args.intersection),
        })

    fig.text(0.5, 0.05, "LRE cosine improvement (Δcos)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, "Hallucination rate (hall / (hall + correct))", rotation=90,
             ha="center", va="center", fontsize=14)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.28)

    base = os.path.join(args.outdir, "scatter_panel_4models_hall_rate_given_value_fit")
    save_figure(fig, base, dpi=args.dpi)
    plt.close(fig)

    corr_path = os.path.join(args.outdir, "correlation_summary_hall_rate_given_value.csv")
    pd.DataFrame(corr_rows).to_csv(corr_path, index=False)
    print(f"[done] wrote: {base}.pdf/.png")
    print(f"[done] wrote: {corr_path}")


if __name__ == "__main__":
    main()
