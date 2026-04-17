#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure-2-like plotting for LRE21 natural (3-way judged) results.

Input:
  --csv: a merged table that contains at least:
         model_key, relation_key, [relation_group], deltacos/delta_cos/...,
         hall_rate_answered, hall_rate_unknown
  --group-map (optional): a CSV that contains relation_key -> relation_group mapping,
         used only if --csv lacks relation_group.

Outputs (written into --outdir):
  - fig2_lre21_nat_3way_scatter_answered_panel_4models.{png,pdf}
  - fig2_lre21_nat_3way_scatter_unknown_panel_4models.{png,pdf}
  - fig2_lre21_nat_3way_corr_like_paper.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.stats import pearsonr  # type: ignore
except Exception:
    pearsonr = None


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

GROUP_ORDER = ["factual", "commonsense", "linguistic", "bias"]
GROUP_TITLE = {
    "factual": "Factual",
    "commonsense": "Commonsense",
    "linguistic": "Linguistic",
    "bias": "Bias",
}


def _pick_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find column for {what}. Tried: {candidates}. Found cols: {list(df.columns)}")


def corr_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan"), float("nan")
    if pearsonr is None:
        return float(np.corrcoef(x, y)[0, 1]), float("nan")
    r, p = pearsonr(x, y)  # two-sided p
    return float(r), float(p)


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def group_styles(groups: List[str]) -> Dict[str, Dict[str, str]]:
    # Use matplotlib default color cycle for "paper-like" look.
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    ordered = [g for g in GROUP_ORDER if g in groups] + [g for g in sorted(groups) if g not in GROUP_ORDER]
    out: Dict[str, Dict[str, str]] = {}
    for i, g in enumerate(ordered):
        out[g] = {"color": colors[i % len(colors)], "marker": markers[i % len(markers)]}
    return out


def plot_panel(
    df: pd.DataFrame,
    outdir: str,
    x_col: str,
    y_col: str,
    y_label: str,
    fname_base: str,
    dpi: int = 300,
) -> None:
    # Global x-range shared by all subplots (same style as the paper script)
    x_all = df[x_col].to_numpy(float)
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    pad = 0.04 * (xmax - xmin) if xmax > xmin else 0.04
    xmin -= pad
    xmax += pad

    y_min, y_max = -0.05, 1.05

    use_group = "relation_group" in df.columns
    if use_group:
        groups = df["relation_group"].astype(str).unique().tolist()
        styles = group_styles(groups)

        legend_handles = [
            Line2D(
                [0], [0],
                marker=styles[g]["marker"],
                color="w",
                label=GROUP_TITLE.get(g, g),
                markerfacecolor=styles[g]["color"],
                markeredgecolor="black",
                markersize=8,
                linewidth=0,
            )
            for g in ([g for g in GROUP_ORDER if g in styles] + [g for g in sorted(styles) if g not in GROUP_ORDER])
        ]
    else:
        styles = {"all": {"color": "C0", "marker": "o"}}
        legend_handles = []

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub[x_col].to_numpy(float)
        y = sub[y_col].to_numpy(float)

        r, p = corr_r_p(x, y)
        slope, intercept = fit_line(x, y)

        if use_group:
            for g, st in styles.items():
                ss = sub[sub["relation_group"].astype(str) == g]
                if ss.empty:
                    continue
                ax.scatter(
                    ss[x_col],
                    ss[y_col],
                    s=75,
                    marker=st["marker"],
                    facecolor=st["color"],
                    edgecolor="black",
                    linewidth=0.75,
                    zorder=3,
                )
        else:
            ax.scatter(
                x,
                y,
                s=75,
                marker=styles["all"]["marker"],
                facecolor=styles["all"]["color"],
                edgecolor="black",
                linewidth=0.75,
                zorder=3,
            )

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(xmin, xmax, 200)
            ax.plot(xx, slope * xx + intercept, linestyle="--", linewidth=1.6, color="black", alpha=0.55, zorder=2)

        title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.4f}, p={p:.4f})" if np.isfinite(p) else f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f})"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.05, "Relational linearity (Δcos)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, y_label, rotation=90, ha="center", va="center", fontsize=14)

    if legend_handles:
        # Put legend inside the bottom-right subplot (Qwen), same as your paper plot.
        leg = axes[3].legend(
            handles=legend_handles,
            title="Relation group",
            loc="lower right",
            frameon=True,
            fontsize=10,
            title_fontsize=11,
            borderaxespad=0.4,
            labelspacing=0.35,
            handletextpad=0.5,
        )
        leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.28)

    base = os.path.join(outdir, fname_base)
    fig.savefig(base + ".png", dpi=dpi)
    fig.savefig(base + ".pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to fig2_lre21_nat_3way_plus_deltacos.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for plots")
    ap.add_argument("--group-map", default="", help="Optional: CSV with relation_key,relation_group (used if missing in --csv)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    need_base = {"model_key", "relation_key", "hall_rate_answered", "hall_rate_unknown"}
    miss = need_base - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in --csv: {sorted(miss)}")

    # Pick Δcos column robustly (your repo has used multiple names historically).
    x_col = _pick_col(
        df,
        candidates=[
            "delta_cos",
            "deltacos",
            "cos_improvement",
            "delta_cos_mean_value",
            "delta_cos_mean",
            "delta_cos_value",
        ],
        what="Δcos",
    )

    # Ensure relation_group exists (for color/marker encoding)
    if "relation_group" not in df.columns:
        if not args.group_map:
            raise ValueError(
                "relation_group is missing in --csv, and --group-map was not provided.\n"
                "Provide --group-map (e.g., exp1_behavior_plus_deltacos.csv) so we can merge relation_group."
            )
        gm = pd.read_csv(args.group_map)
        if "relation_key" not in gm.columns or "relation_group" not in gm.columns:
            raise ValueError(f"--group-map must contain columns relation_key and relation_group. Found: {list(gm.columns)}")
        gm = gm[["relation_key", "relation_group"]].drop_duplicates()
        df = df.merge(gm, on="relation_key", how="left")
        if df["relation_group"].isna().any():
            missing_rel = sorted(df[df["relation_group"].isna()]["relation_key"].astype(str).unique().tolist())
            raise ValueError(f"Some relations still missing relation_group after merge: {missing_rel}")

    # Keep only the four models (and stable ordering)
    df["model_key"] = df["model_key"].astype(str)
    df = df[df["model_key"].isin(MODEL_ORDER)].copy()

    # Write a small corr summary (useful for rebuttal logs)
    rows = []
    for y_col in ["hall_rate_answered", "hall_rate_unknown"]:
        for mk in MODEL_ORDER:
            sub = df[df["model_key"] == mk]
            r, p = corr_r_p(sub[x_col].to_numpy(float), sub[y_col].to_numpy(float))
            rows.append({"metric": y_col, "model_key": mk, "n_rel": int(len(sub)), "r": r, "p_two": p})
    corr_path = os.path.join(args.outdir, "fig2_lre21_nat_3way_corr_like_paper.csv")
    pd.DataFrame(rows).to_csv(corr_path, index=False)
    print(f"[write] {corr_path}")

    # Panel 1: answered = hall/(hall+correct)
    plot_panel(
        df=df,
        outdir=args.outdir,
        x_col=x_col,
        y_col="hall_rate_answered",
        y_label="Hallucination rate among answered cases",
        fname_base="fig2_lre21_nat_3way_scatter_answered_panel_4models",
        dpi=args.dpi,
    )
    print(f"[plot] wrote: {os.path.join(args.outdir, 'fig2_lre21_nat_3way_scatter_answered_panel_4models.pdf')}")

    # Panel 2: unknown = hall/(hall+refusal)
    plot_panel(
        df=df,
        outdir=args.outdir,
        x_col=x_col,
        y_col="hall_rate_unknown",
        y_label="Hallucination rate (Hall/(Hall+Refusal))",
        fname_base="fig2_lre21_nat_3way_scatter_unknown_panel_4models",
        dpi=args.dpi,
    )
    print(f"[plot] wrote: {os.path.join(args.outdir, 'fig2_lre21_nat_3way_scatter_unknown_panel_4models.pdf')}")

    if pearsonr is None:
        print("[WARN] SciPy not found; p-values will be NaN. Install via: pip install scipy")


if __name__ == "__main__":
    main()
