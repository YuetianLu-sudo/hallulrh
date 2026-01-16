#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------
# Pretty names / ordering
# ---------------------------

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

REL_ORDER = [
    "father",
    "instrument",
    "sport",
    "company_ceo",
    "company_hq",
    "country_language",
]

REL_LABEL = {
    "father": "father (first name)",
    "instrument": "instrument",
    "sport": "sport",
    "company_ceo": "company CEO",
    "company_hq": "company HQ",
    "country_language": "country language",
}


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _relation_style_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (color_map, marker_map) with a stable mapping across plots."""
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    color_map = {rel: colors[i % len(colors)] for i, rel in enumerate(REL_ORDER)}
    marker_list = ["o", "s", "^", "D", "P", "X"]  # 6 relations
    marker_map = {rel: marker_list[i % len(marker_list)] for i, rel in enumerate(REL_ORDER)}
    return color_map, marker_map


def save_figure(fig: plt.Figure, base_path_no_ext: str, dpi: int = 300, bbox_inches=None) -> None:
    """Save both PNG (for slides) and PDF (vector, for papers)."""
    png_path = base_path_no_ext + ".png"
    pdf_path = base_path_no_ext + ".pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(pdf_path, bbox_inches=bbox_inches)


# ---------------------------
# Plot 1: Behavior bars (horizontal, stacked)
# ---------------------------

def plot_behavior_bars(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """
    Expects columns:
      - model_key
      - relation
      - halluc_rate
      - refusal_rate
      - halluc_ci_low/high  (optional)
      - refusal_ci_low/high (optional)
    """
    os.makedirs(outdir, exist_ok=True)

    for mk in MODEL_ORDER:
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            continue

        sub["relation"] = pd.Categorical(sub["relation"], categories=REL_ORDER, ordered=True)
        sub = sub.sort_values("relation")

        rels = [REL_LABEL.get(r, r) for r in sub["relation"].astype(str).tolist()]
        y = np.arange(len(rels))

        halluc = sub["halluc_rate"].to_numpy(dtype=float)
        refusal = sub["refusal_rate"].to_numpy(dtype=float)

        halluc_err = None
        refusal_err = None
        if {"halluc_ci_low", "halluc_ci_high"}.issubset(sub.columns):
            h_low = sub["halluc_ci_low"].to_numpy(dtype=float)
            h_high = sub["halluc_ci_high"].to_numpy(dtype=float)
            halluc_err = np.vstack([halluc - h_low, h_high - halluc])

        if {"refusal_ci_low", "refusal_ci_high"}.issubset(sub.columns):
            r_low = sub["refusal_ci_low"].to_numpy(dtype=float)
            r_high = sub["refusal_ci_high"].to_numpy(dtype=float)
            refusal_err = np.vstack([refusal - r_low, r_high - refusal])

        fig, ax = plt.subplots(figsize=(7.8, 3.6))

        ax.barh(
            y,
            halluc,
            xerr=halluc_err,
            capsize=3,
            label="Hallucination",
            alpha=0.95,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.barh(
            y,
            refusal,
            left=halluc,
            xerr=refusal_err,
            capsize=3,
            label="Refusal",
            alpha=0.60,
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(rels, fontsize=11)
        ax.invert_yaxis()

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Rate", fontsize=12)
        ax.set_title(f"{MODEL_TITLE.get(mk, mk)}: refusal vs hallucination", fontsize=13)

        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

        ax.legend(loc="lower right", frameon=True, fontsize=10)

        fig.tight_layout()
        base = os.path.join(outdir, f"{mk}_behavior_pub")
        save_figure(fig, base, dpi=dpi)
        plt.close(fig)


# ---------------------------
# Plot 2: Scatter (legend-based, no text labels)
# ---------------------------

def plot_scatter_per_model(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """
    Per-model scatter: x = cos_improvement, y = halluc_rate.
    Uses relation-coded points + legend (no text labels) to avoid overlaps.
    """
    os.makedirs(outdir, exist_ok=True)
    color_map, marker_map = _relation_style_maps()

    for mk in MODEL_ORDER:
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            continue
        sub = sub.set_index("relation").reindex(REL_ORDER).reset_index()

        x = sub["cos_improvement"].to_numpy(dtype=float)
        y = sub["halluc_rate"].to_numpy(dtype=float)
        r = pearson_r(x, y)

        fig, ax = plt.subplots(figsize=(5.4, 4.1))

        for rel in REL_ORDER:
            row = sub[sub["relation"] == rel]
            if row.empty:
                continue
            ax.scatter(
                row["cos_improvement"].iat[0],
                row["halluc_rate"].iat[0],
                s=120,
                marker=marker_map[rel],
                facecolor=color_map[rel],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                clip_on=False,
            )

        ax.set_title(f"{MODEL_TITLE.get(mk, mk)} (r={r:.2f})", fontsize=13)
        ax.set_xlabel("LRE cosine improvement (Δcos)", fontsize=12)
        ax.set_ylabel("Hallucination rate", fontsize=12)

        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6))

        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

        handles = [
            Line2D(
                [0], [0],
                marker=marker_map[rel],
                color="w",
                label=REL_LABEL.get(rel, rel),
                markerfacecolor=color_map[rel],
                markeredgecolor="black",
                markersize=9,
                linewidth=0,
            )
            for rel in REL_ORDER
        ]
        ax.legend(
            handles=handles,
            title="Relation",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=10,
            title_fontsize=11,
        )

        fig.tight_layout()
        base = os.path.join(outdir, f"{mk}_lre_vs_halluc_scatter_pub")
        save_figure(fig, base, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_scatter_panel_4models(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """
    2x2 panel with one subplot per model.
    Fixes:
      - No text labels (legend instead) -> no overlaps
      - y-limits beyond [0,1] + clip_on=False -> no marker clipping
      - legend below x-label -> no overlap
    """
    os.makedirs(outdir, exist_ok=True)
    color_map, marker_map = _relation_style_maps()

    global_xmin = float(df["cos_improvement"].min()) - 0.04
    global_xmax = float(df["cos_improvement"].max()) + 0.04
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_key"] == mk].copy()
        sub = sub.set_index("relation").reindex(REL_ORDER).reset_index()

        for rel in REL_ORDER:
            row = sub[sub["relation"] == rel]
            if row.empty:
                continue
            ax.scatter(
                row["cos_improvement"].iat[0],
                row["halluc_rate"].iat[0],
                s=110,
                marker=marker_map[rel],
                facecolor=color_map[rel],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                clip_on=False,
            )

        r = pearson_r(
            sub["cos_improvement"].to_numpy(dtype=float),
            sub["halluc_rate"].to_numpy(dtype=float),
        )
        ax.set_title(f"{MODEL_TITLE.get(mk, mk)} (r={r:.2f})", fontsize=13)

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))

        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.19, "LRE cosine improvement (Δcos)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, "Hallucination rate", rotation=90, ha="center", va="center", fontsize=14)

    handles = [
        Line2D(
            [0], [0],
            marker=marker_map[rel],
            color="w",
            label=REL_LABEL.get(rel, rel),
            markerfacecolor=color_map[rel],
            markeredgecolor="black",
            markersize=9,
            linewidth=0,
        )
        for rel in REL_ORDER
    ]

    legend = fig.legend(
        handles=handles,
        labels=[REL_LABEL.get(rel, rel) for rel in REL_ORDER],
        title="Relation",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.30, wspace=0.18, hspace=0.28)

    base = os.path.join(outdir, "scatter_panel_4models_pub")
    save_figure(fig, base, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to behavior_plus_lre.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for publication plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG outputs (default: 300)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {"model_key", "relation", "halluc_rate", "refusal_rate", "cos_improvement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    plot_behavior_bars(df, outdir=outdir, dpi=args.dpi)
    plot_scatter_per_model(df, outdir=outdir, dpi=args.dpi)
    plot_scatter_panel_4models(df, outdir=outdir, dpi=args.dpi)

    print(f"[pubplots] Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()
