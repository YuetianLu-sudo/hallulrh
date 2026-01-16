#!/usr/bin/env python3
"""Publication-ready plots for LRE vs behavior.

This version avoids per-point text annotations (which can overlap) by encoding
relations via color + marker and using a legend.

Inputs: a joined CSV with at least:
  - model_key
  - relation
  - refusal_rate
  - halluc_rate
  - cos_improvement

Outputs:
  - per-model behavior bar plots
  - per-model scatter plots (Δcos vs hallucination rate)
  - a 2x2 panel scatter plot
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -----------------------
# Global plotting style
# -----------------------
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

MODEL_NAME_MAP = {
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

# Distinct marker per relation so points remain distinguishable even if overlapping.
REL_MARKER = {
    "father": "o",
    "instrument": "s",
    "sport": "^",
    "company_ceo": "D",
    "company_hq": "P",
    "country_language": "X",
}


def _rel_color(rel: str):
    """Stable relation->color mapping."""
    cmap = plt.get_cmap("tab10")
    try:
        idx = REL_ORDER.index(rel)
    except ValueError:
        idx = 0
    return cmap(idx % 10)


def _sorted_relations(series: pd.Series) -> pd.Series:
    order = {r: i for i, r in enumerate(REL_ORDER)}
    return series.map(lambda r: order.get(r, 999))


def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _legend_handles() -> List[Line2D]:
    handles: List[Line2D] = []
    for rel in REL_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=REL_MARKER.get(rel, "o"),
                linestyle="None",
                markersize=8,
                markerfacecolor=_rel_color(rel),
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=REL_LABEL.get(rel, rel),
            )
        )
    return handles


def _save_fig(fig: plt.Figure, out_base: str) -> None:
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_behavior_bars(df_model: pd.DataFrame, model_key: str, out_dir: str) -> None:
    df = df_model.copy()
    df["relation"] = df["relation"].astype(str)
    df = df.sort_values("relation", key=_sorted_relations)

    rels = df["relation"].tolist()
    refusals = df["refusal_rate"].to_numpy(dtype=float)
    halluc = df["halluc_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    x = np.arange(len(rels))
    ax.bar(x, refusals, label="refusal")
    ax.bar(x, halluc, bottom=refusals, label="hallucination")

    ax.set_xticks(x)
    ax.set_xticklabels([REL_LABEL.get(r, r) for r in rels], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    title = MODEL_NAME_MAP.get(model_key, model_key)
    ax.set_title(f"{title}: refusal vs hallucination (Gemini judge)")
    ax.legend(frameon=False)

    base = os.path.join(out_dir, f"{model_key}_behavior_bars_pub")
    _save_fig(fig, base)


def _axis_limits_global(df: pd.DataFrame) -> Dict[str, float]:
    x = df["cos_improvement"].to_numpy(dtype=float)
    y = df["halluc_rate"].to_numpy(dtype=float)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    # Add small padding, and clamp y within [0, 1].
    x_pad = 0.04 if x_max - x_min < 1e-6 else 0.06 * (x_max - x_min)
    y_pad = 0.03

    return {
        "x_min": x_min - x_pad,
        "x_max": x_max + x_pad,
        "y_min": max(0.0, y_min - y_pad),
        "y_max": min(1.0, y_max + y_pad),
    }


def _plot_scatter_model(df_model: pd.DataFrame, model_key: str, out_dir: str, lims: Dict[str, float]) -> None:
    df = df_model.copy()
    df["relation"] = df["relation"].astype(str)
    df = df.sort_values("relation", key=_sorted_relations)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))

    # Optional error bars if CI columns exist.
    has_ci = all(c in df.columns for c in ["halluc_ci_low", "halluc_ci_high"])
    if has_ci:
        for row in df.itertuples(index=False):
            try:
                yerr_low = float(row.halluc_rate) - float(row.halluc_ci_low)
                yerr_high = float(row.halluc_ci_high) - float(row.halluc_rate)
                ax.errorbar(
                    float(row.cos_improvement),
                    float(row.halluc_rate),
                    yerr=[[max(0.0, yerr_low)], [max(0.0, yerr_high)]],
                    fmt="none",
                    elinewidth=1.0,
                    capsize=2.5,
                    alpha=0.35,
                    zorder=1,
                )
            except Exception:
                pass

    # Draw points per relation (color + marker), no text labels.
    for rel in REL_ORDER:
        sub = df[df["relation"] == rel]
        if sub.empty:
            continue
        ax.scatter(
            sub["cos_improvement"].to_numpy(dtype=float),
            sub["halluc_rate"].to_numpy(dtype=float),
            s=95,
            marker=REL_MARKER.get(rel, "o"),
            color=_rel_color(rel),
            edgecolors="black",
            linewidths=0.8,
            alpha=0.92,
            zorder=2,
            label=REL_LABEL.get(rel, rel),
        )

    # Pearson r for title (with very few points, keep simple).
    x = df["cos_improvement"].to_numpy(dtype=float)
    y = df["halluc_rate"].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")

    title = MODEL_NAME_MAP.get(model_key, model_key)
    ax.set_title(f"{title}: Δcos vs hallucination (Pearson r={r:.2f})")

    ax.set_xlabel("Δcos improvement")
    ax.set_ylabel("Hallucination rate")
    ax.grid(True, alpha=0.25, linewidth=0.7)

    ax.set_xlim(lims["x_min"], lims["x_max"])
    ax.set_ylim(lims["y_min"], lims["y_max"])

    ax.legend(
        title="Relation",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    base = os.path.join(out_dir, f"{model_key}_lre_vs_halluc_scatter_pub")
    _save_fig(fig, base)


def _plot_scatter_panel(df: pd.DataFrame, out_dir: str, lims: Dict[str, float]) -> None:
    keys = [k for k in MODEL_NAME_MAP.keys() if k in set(df["model_key"].astype(str).tolist())]
    if not keys:
        keys = sorted(df["model_key"].astype(str).unique().tolist())

    # Keep a stable order if possible.
    preferred = ["gemma_7b_it", "llama3_1_8b_instruct", "mistral_7b_instruct", "qwen2_5_7b_instruct"]
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    keys = keys[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, model_key in zip(axes, keys):
        sub = df[df["model_key"].astype(str) == model_key].copy()
        sub["relation"] = sub["relation"].astype(str)
        sub = sub.sort_values("relation", key=_sorted_relations)

        for rel in REL_ORDER:
            rsub = sub[sub["relation"] == rel]
            if rsub.empty:
                continue
            ax.scatter(
                rsub["cos_improvement"].to_numpy(dtype=float),
                rsub["halluc_rate"].to_numpy(dtype=float),
                s=70,
                marker=REL_MARKER.get(rel, "o"),
                color=_rel_color(rel),
                edgecolors="black",
                linewidths=0.7,
                alpha=0.92,
                zorder=2,
            )

        x = sub["cos_improvement"].to_numpy(dtype=float)
        y = sub["halluc_rate"].to_numpy(dtype=float)
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
        ax.set_title(f"{MODEL_NAME_MAP.get(model_key, model_key)} (r={r:.2f})")

        ax.grid(True, alpha=0.25, linewidth=0.7)

    # Hide any unused axes.
    for j in range(len(keys), 4):
        axes[j].axis("off")

    for ax in axes:
        ax.set_xlim(lims["x_min"], lims["x_max"])
        ax.set_ylim(lims["y_min"], lims["y_max"])

    fig.supxlabel("Δcos improvement")
    fig.supylabel("Hallucination rate")

    handles = _legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        title="Relation",
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0.02, 0.06, 1.0, 0.98])

    base = os.path.join(out_dir, "panel_lre_vs_halluc_scatter_pub")
    _save_fig(fig, base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="Joined CSV (judge summary + LRE).")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required = {"model_key", "relation", "refusal_rate", "halluc_rate", "cos_improvement"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise SystemExit(f"[error] Missing required columns: {missing}")

    df["model_key"] = df["model_key"].astype(str)
    df["relation"] = df["relation"].astype(str)

    _ensure_out_dir(args.out_dir)

    lims = _axis_limits_global(df)

    # Per-model plots
    for model_key, sub in df.groupby("model_key", sort=False):
        _plot_behavior_bars(sub, model_key, args.out_dir)
        _plot_scatter_model(sub, model_key, args.out_dir, lims)

    # Panel plot
    _plot_scatter_panel(df, args.out_dir, lims)

    print(f"[ok] Wrote pub plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
