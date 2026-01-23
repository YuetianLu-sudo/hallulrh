#!/usr/bin/env python3
"""
Plot Figure-1-style scatter panel for Exp1 (factual-only):
x = delta_cos (LRE), y = hall_rate = hall/(hall+ref) on synthetic prompts.

Outputs:
- exp1_scatter_panel_4models_factual.pdf/png  (2x2 panel, paper-ready)
- exp1_<model>_labeled.pdf (optional, debugging with relation labels)
- exp1_corr_summary_factual.csv (numbers to paste into rebuttal)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress


MODEL_ORDER = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

MODEL_TITLE = {
    "gemma_7b_it": "Gemma-7B-IT",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "Mistral-7B-Instruct-v0.3",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
}


def weighted_pearsonr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation with nonnegative weights."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()

    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))


def plot_panel(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> pd.DataFrame:
    # Determine models present
    models = [m for m in MODEL_ORDER if m in set(df["model_name"].unique())]
    if len(models) != 4:
        # Fallback: deterministic ordering
        models = sorted(df["model_name"].unique().tolist())

    # Global x range for comparability
    x_min = float(df["delta_cos"].min())
    x_max = float(df["delta_cos"].max())
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    x_grid = np.linspace(x_min - pad, x_max + pad, 200)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5), sharex=True, sharey=True)
    axes = axes.flatten()

    corr_rows = []

    for ax, m in zip(axes, models):
        sub = df[df["model_name"] == m].copy()

        x = sub["delta_cos"].to_numpy(dtype=float)
        y = sub["hall_rate"].to_numpy(dtype=float)

        # Scatter
        ax.scatter(x, y)

        # OLS line for visualization (same spirit as main paper)
        lr = linregress(x, y)
        ax.plot(x_grid, lr.intercept + lr.slope * x_grid, linewidth=1.0)

        # Correlations
        r, p = pearsonr(x, y)
        rho, p_rho = spearmanr(x, y)

        # Weighted r (weights = lre_n if present)
        if "lre_n" in sub.columns:
            rw = weighted_pearsonr(x, y, sub["lre_n"].to_numpy(dtype=float))
        else:
            rw = float("nan")

        corr_rows.append(
            {
                "model_name": m,
                "pearson_r": r,
                "pearson_p": p,
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "r_weighted_by_lre_n": rw,
                "n_relations": len(sub),
            }
        )

        title = MODEL_TITLE.get(m, m)
        ax.set_title(f"{title}\nPearson r={r:.3f}")

        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.02, 1.02)

    # Axis labels (Figure-1-style)
    axes[0].set_ylabel("Hallucination rate\n(hall/(hall+ref))")
    axes[2].set_ylabel("Hallucination rate\n(hall/(hall+ref))")
    axes[2].set_xlabel("Relational linearity (Δcos)")
    axes[3].set_xlabel("Relational linearity (Δcos)")

    # Hide duplicate y tick labels on right column
    axes[1].yaxis.set_tick_params(labelleft=False)
    axes[3].yaxis.set_tick_params(labelleft=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(corr_rows)


def plot_labeled_per_model(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in sorted(df["model_name"].unique().tolist()):
        sub = df[df["model_name"] == m].copy()
        x = sub["delta_cos"].to_numpy(float)
        y = sub["hall_rate"].to_numpy(float)
        rel = sub["relation"].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.scatter(x, y)
        for xx, yy, rr in zip(x, y, rel):
            ax.annotate(rr, (xx, yy), xytext=(5, 3), textcoords="offset points", fontsize=8)

        r, p = pearsonr(x, y)
        ax.set_title(f"{MODEL_TITLE.get(m, m)} (labeled)\nPearson r={r:.3f}, p={p:.3g}")
        ax.set_xlabel("Relational linearity (Δcos)")
        ax.set_ylabel("Hallucination rate (hall/(hall+ref))")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(out_dir / f"exp1_{m}_labeled.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns: model_name, relation, hall_rate, delta_cos, relation_group, lre_n (optional).")
    ap.add_argument("--out-dir", required=True, help="Directory to write figures and summary.")
    ap.add_argument("--factual-only", action="store_true", help="Filter relation_group == 'factual'. Recommended.")
    ap.add_argument("--make-labeled", action="store_true", help="Also write per-model labeled scatter PDFs for debugging.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    if args.factual_only:
        if "relation_group" not in df.columns:
            raise RuntimeError("Asked for --factual-only but column relation_group is missing.")
        df = df[df["relation_group"] == "factual"].copy()

    # Sanity checks
    required = ["model_name", "relation", "hall_rate", "delta_cos"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    out_pdf = out_dir / "exp1_scatter_panel_4models_factual.pdf"
    out_png = out_dir / "exp1_scatter_panel_4models_factual.png"
    corr_df = plot_panel(df, out_pdf, out_png)

    corr_path = out_dir / "exp1_corr_summary_factual.csv"
    corr_df.to_csv(corr_path, index=False)

    print("[done] wrote:")
    print("  -", out_pdf)
    print("  -", out_png)
    print("  -", corr_path)
    print()
    print(corr_df.to_string(index=False))

    if args.make_labeled:
        labeled_dir = out_dir / "labeled_per_model"
        plot_labeled_per_model(df, labeled_dir)
        print("[done] labeled per-model PDFs in:", labeled_dir)


if __name__ == "__main__":
    main()
