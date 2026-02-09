#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure2-style dashboard (2x2) for LRE natural eval:
  y = hallucination / (hallucination + correct)  == Hall|Value
  x = Δcos  (here: cos_improvement)

This script enforces the same style as the paper Figure 2:
- color/marker by relation_group (Factual/Commonsense/Linguistic/Bias)
- dashed linear fit
- per-panel Pearson r and two-sided p
- legend inside bottom-right panel

New features:
- --min-n-test: filter relations by Δcos reliability (keep rows with n_test >= threshold).
  Use --min-n-test 11 to match "n_test > 10" (gt10 / min11).
- --tag: output filename tag (e.g., all47 / gt10) to avoid overwriting.
- --no-intersection: optional; by default we use the intersection of relations across models after filtering.
"""

import argparse
import json
import os
from typing import Dict, List, Set, Tuple

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

MARKER_MAP = {
    "factual": "o",
    "commonsense": "s",
    "linguistic": "^",
    "bias": "D",
}


def load_rel2group(prompts_path: str) -> Dict[str, str]:
    rel2group: Dict[str, str] = {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rk = rec.get("relation_key")
            grp = rec.get("relation_group")
            if isinstance(rk, str) and rk and isinstance(grp, str) and grp and rk not in rel2group:
                rel2group[rk] = grp
    return rel2group


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


def filter_by_min_n_test(df: pd.DataFrame, min_n_test: int) -> pd.DataFrame:
    if min_n_test <= 0:
        return df
    if "n_test" not in df.columns:
        raise ValueError("CSV missing required column 'n_test' for --min-n-test filtering.")
    out = df.copy()
    out["n_test"] = pd.to_numeric(out["n_test"], errors="coerce")
    out = out[out["n_test"] >= float(min_n_test)].copy()
    return out


def intersect_relations(df: pd.DataFrame, models: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    rel_sets: List[Set[str]] = []
    for mk in models:
        ss = df[df["model_key"].astype(str) == mk]["relation_key"].astype(str).tolist()
        rel_sets.append(set(ss))
    inter = set.intersection(*rel_sets) if rel_sets else set()
    inter_sorted = sorted(inter)
    df2 = df[df["relation_key"].astype(str).isin(inter_sorted)].copy()
    return df2, inter_sorted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="lre3way_behavior_plus_deltacos_all47.csv")
    ap.add_argument("--prompts", required=True, help="lre_prompts_qonly.jsonl (for relation_group mapping)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--ycol", default="hall_rate_answered",
                    help="Hall|Value column (default: hall_rate_answered)")
    ap.add_argument("--min-n-test", type=int, default=0,
                    help="Keep relations with n_test >= this value. Use 11 to implement 'n_test > 10'.")
    ap.add_argument("--tag", default=None,
                    help="Output filename tag (e.g., all47, gt10). If omitted, auto-derives from min-n-test.")
    ap.add_argument("--no-intersection", action="store_true",
                    help="If set, do NOT take intersection of relations across models after filtering.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = {"model_key", "relation_key", "cos_improvement", args.ycol}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in CSV: {sorted(miss)}")

    # Add relation_group if missing
    if "relation_group" not in df.columns:
        rel2group = load_rel2group(args.prompts)
        df["relation_group"] = df["relation_key"].astype(str).map(rel2group)

    missing_group = df["relation_group"].isna()
    if int(missing_group.sum()) > 0:
        bad = sorted(df.loc[missing_group, "relation_key"].astype(str).unique().tolist())
        raise RuntimeError(
            f"Found {missing_group.sum()} rows with missing relation_group. "
            f"Missing relation_keys (unique)={bad[:20]}{'...' if len(bad) > 20 else ''}"
        )

    # Filter by min_n_test (Δcos reliability)
    df_f = filter_by_min_n_test(df, args.min_n_test)

    # By default, take the intersection across models after filtering
    if not args.no_intersection:
        df_f, inter = intersect_relations(df_f, MODEL_ORDER)
        print(f"[filter] min_n_test={args.min_n_test} -> intersection relations={len(inter)}")
    else:
        inter = sorted(df_f["relation_key"].astype(str).unique().tolist())
        print(f"[filter] min_n_test={args.min_n_test} -> no intersection; unique relations={len(inter)}")

    # Fixed group styles (match Figure2: C0/C1/C2/C3)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4"]

    styles = {}
    for i, g in enumerate(GROUP_ORDER):
        styles[g] = {"color": colors[i % len(colors)], "marker": MARKER_MAP[g]}

    legend_handles = [
        Line2D(
            [0], [0],
            marker=styles[g]["marker"], color="w",
            label=GROUP_TITLE.get(g, g),
            markerfacecolor=styles[g]["color"],
            markeredgecolor="black",
            markersize=8, linewidth=0
        )
        for g in GROUP_ORDER
    ]

    # Global axis range
    x_all = df_f["cos_improvement"].to_numpy(float)
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    pad = 0.04 * (xmax - xmin) if xmax > xmin else 0.04
    xmin -= pad
    xmax += pad
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df_f[df_f["model_key"].astype(str) == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub["cos_improvement"].to_numpy(float)
        y = sub[args.ycol].to_numpy(float)

        r, p = corr_r_p(x, y)
        slope, intercept = fit_line(x, y)

        for g in GROUP_ORDER:
            ss = sub[sub["relation_group"].astype(str) == g]
            if ss.empty:
                continue
            ax.scatter(
                ss["cos_improvement"], ss[args.ycol],
                s=75,
                marker=styles[g]["marker"],
                facecolor=styles[g]["color"],
                edgecolor="black",
                linewidth=0.75,
                zorder=3,
            )

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(xmin, xmax, 200)
            ax.plot(
                xx, slope * xx + intercept,
                linestyle="--",
                linewidth=1.6,
                color="black",
                alpha=0.55,
                zorder=2,
            )

        title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.4f}, p={p:.4f})" if np.isfinite(p) else f"{MODEL_TITLE.get(mk, mk)} (r={r:.4f})"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))

        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.05, "Δcos (mean over value-providing triples)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, "Hallucination | Value (Hall|Value)", rotation=90, ha="center", va="center", fontsize=14)

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

    tag = args.tag
    if not tag:
        tag = "all" if args.min_n_test <= 0 else f"min{args.min_n_test}"

    base = os.path.join(args.outdir, f"scatter_panel_4models_hall_over_value_{tag}_fit_fig2style")
    fig.savefig(base + ".png", dpi=args.dpi)
    fig.savefig(base + ".pdf")
    plt.close(fig)

    print("[done] wrote:", base + ".pdf")
    print("[done] wrote:", base + ".png")
    print(f"[done] points per model:")
    for mk in MODEL_ORDER:
        n = int((df_f["model_key"].astype(str) == mk).sum())
        print(f"  - {mk}: {n}")

    if pearsonr is None:
        print("[WARN] SciPy not found -> p-values are NaN. Install via: pip install scipy")


if __name__ == "__main__":
    main()
