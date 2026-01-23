#!/usr/bin/env python3
"""
EXP1 (factual-only) Figure-1-style scatter panel.

Input CSV must contain:
  - model_name
  - relation
  - hall_rate   (hall/(hall+ref))
  - delta_cos

Outputs:
  - exp1_factual_scatter_panel_4models_fit.(pdf|png)
  - optional annotated debug version with --annotate
"""
import argparse
import os
from typing import List, Tuple

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

def pearson_r_p_two(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan"), float("nan")
    try:
        import scipy.stats as st  # type: ignore
        r, p = st.pearsonr(x, y)  # two-sided p
        return float(r), float(p)
    except Exception:
        # Fallback without SciPy: r only
        r = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 else float("nan")
        return r, float("nan")

def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # Least-squares line: y = slope*x + intercept
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan"), float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)

def save_figure(fig: plt.Figure, base_path_no_ext: str, dpi: int = 300) -> None:
    fig.savefig(base_path_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path_no_ext + ".pdf", bbox_inches="tight")

def _short_rel_label(rel: str) -> str:
    # Compact label for optional annotation
    return rel.replace("_", "\\n")

def plot_panel(df: pd.DataFrame, outdir: str, dpi: int = 300, annotate: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Stable relation order: sort by mean delta_cos across models
    rel_order: List[str] = (
        df.groupby("relation")["delta_cos"].mean().sort_values().index.astype(str).tolist()
    )

    global_xmin = float(df["delta_cos"].min()) - 0.04
    global_xmax = float(df["delta_cos"].max()) + 0.04
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_name"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub["relation"] = pd.Categorical(sub["relation"], categories=rel_order, ordered=True)
        sub = sub.sort_values("relation")

        x = sub["delta_cos"].to_numpy(dtype=float)
        y = sub["hall_rate"].to_numpy(dtype=float)

        r, p_two = pearson_r_p_two(x, y)
        slope, intercept = fit_line(x, y)

        # Points: each point is a relation
        ax.scatter(
            x, y,
            s=90,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            clip_on=False,
        )

        # Optional debug annotation
        if annotate:
            for _, row in sub.iterrows():
                ax.text(
                    float(row["delta_cos"]),
                    float(row["hall_rate"]),
                    _short_rel_label(str(row["relation"])),
                    fontsize=7.5,
                    ha="center",
                    va="center",
                    zorder=4,
                )

        # Fit line (least squares)
        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(global_xmin, global_xmax, 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, linestyle="--", linewidth=1.6, color="black", alpha=0.55, zorder=2)

        if np.isfinite(p_two):
            ax.set_title(f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f}, p={p_two:.4f})", fontsize=13)
        else:
            ax.set_title(f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f})", fontsize=13)

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))

        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.04, "Relational linearity (Δcos)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, "Hallucination rate (hall/(hall+ref))", rotation=90, ha="center", va="center", fontsize=14)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.28)

    base = os.path.join(outdir, "exp1_factual_scatter_panel_4models_fit")
    if annotate:
        base = os.path.join(outdir, "exp1_factual_scatter_panel_4models_fit_annotated")
    save_figure(fig, base, dpi=dpi)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to exp1_behavior_plus_deltacos_factual.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--annotate", action="store_true", help="Annotate points with relation names (debug-only).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    needed = {"model_name", "relation", "hall_rate", "delta_cos"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    plot_panel(df, outdir=args.outdir, dpi=args.dpi, annotate=bool(args.annotate))
    print(f"[plots] Wrote plots to: {args.outdir}")

if __name__ == "__main__":
    main()
