#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None

MODEL_ORDER = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

TITLE_MAP = {
    "gemma_7b_it": "Gemma-7B-IT",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "Mistral-7B-Instruct",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_pdf", required=True)
    ap.add_argument("--filter_group", default="__ALL__", help="If set (e.g., factual), filter relation_group == this value.")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.filter_group != "__ALL__":
        df = df[df["relation_group"] == args.filter_group].copy()

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for ax, m in zip(axes, MODEL_ORDER):
        sub = df[df["model_name"] == m].dropna(subset=["delta_cos", "hall_rate"]).copy()
        x = sub["delta_cos"].to_numpy()
        y = sub["hall_rate"].to_numpy()

        # English-only comment: plot by relation_group for readability (3 groups: factual/commonsense/bias)
        for g, gdf in sub.groupby("relation_group", sort=True):
            h = ax.scatter(gdf["delta_cos"], gdf["hall_rate"], alpha=0.85, s=26, label=g)

        # Fit line
        if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
            a, b = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ys = a * xs + b
            ax.plot(xs, ys, linewidth=1)

        # Corr stats
        if pearsonr is not None and len(x) >= 2:
            r, p = pearsonr(x, y)
            title = f"{TITLE_MAP.get(m, m)}  (r={r:.3f}, p={p:.2g})"
        else:
            r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
            title = f"{TITLE_MAP.get(m, m)}  (r={r:.3f})"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Relational linearity (Δcos)")
        ax.set_ylabel("Hallucination rate  hall/(hall+ref)")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # One shared legend
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(set(legend_labels)), frameon=False)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    print("[done] wrote:", args.output_pdf)

if __name__ == "__main__":
    main()
