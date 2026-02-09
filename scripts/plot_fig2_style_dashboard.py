#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Figure2-style 2x2 dashboard for LRE natural eval.

x = cos_improvement  (Δcos)
y = any rate column (e.g., hall_rate_answered == Hall|Value, hall_rate_noncorrect == Hall/(Hall+Refusal))

Optional:
- filter by --min-n-test (on n_test)
- enforce intersection relations across the 4 models AFTER filtering & finite x/y
"""

import argparse
import json
import os
from typing import Dict, List

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


def corr_r_p(x: np.ndarray, y: np.ndarray):
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


def fit_line(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def ensure_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure hall_rate_answered (= hall/(hall+correct)) and hall_rate_noncorrect (= hall/(hall+refusal)) exist when possible.
    Assumes columns: HALLUCINATION, CORRECT, REFUSAL (as produced by your 3-way aggregation).
    """
    df = df.copy()
    if {"HALLUCINATION", "CORRECT"}.issubset(df.columns) and "hall_rate_answered" not in df.columns:
        denom = (df["HALLUCINATION"].astype(float) + df["CORRECT"].astype(float))
        df["hall_rate_answered"] = np.where(denom > 0, df["HALLUCINATION"].astype(float) / denom, np.nan)

    if {"HALLUCINATION", "REFUSAL"}.issubset(df.columns) and "hall_rate_noncorrect" not in df.columns:
        denom = (df["HALLUCINATION"].astype(float) + df["REFUSAL"].astype(float))
        df["hall_rate_noncorrect"] = np.where(denom > 0, df["HALLUCINATION"].astype(float) / denom, np.nan)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="e.g., lre3way_behavior_plus_deltacos_all47.csv")
    ap.add_argument("--prompts", required=True, help="lre_prompts_qonly.jsonl (for relation_group mapping)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--ycol", required=True, help="e.g., hall_rate_answered or hall_rate_noncorrect")
    ap.add_argument("--ylabel", required=True, help="y-axis label text")

    ap.add_argument("--min-n-test", type=int, default=None, help="Filter by n_test >= this value")
    ap.add_argument("--enforce-intersection", action="store_true",
                    help="Keep only relations present for all 4 models after filtering & finite x/y")
    ap.add_argument("--tag", default="", help="Optional suffix for output filenames")
    ap.add_argument("--title-p-decimals", type=int, default=4)
    ap.add_argument("--title-r-decimals", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = ensure_rate_columns(df)

    # Basic checks
    need = {"model_key", "relation_key", "cos_improvement", args.ycol}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in CSV: {sorted(miss)}")

    # Filter by n_test
    if args.min_n_test is not None:
        if "n_test" not in df.columns:
            raise ValueError("CSV missing n_test; cannot apply --min-n-test.")
        df = df[df["n_test"].astype(float) >= float(args.min_n_test)].copy()

    # Add relation_group if missing / partially missing
    if "relation_group" not in df.columns or df["relation_group"].isna().any():
        rel2group = load_rel2group(args.prompts)
        df["relation_group"] = df["relation_key"].astype(str).map(rel2group)

    # Drop rows missing relation_group
    if df["relation_group"].isna().any():
        bad = sorted(df.loc[df["relation_group"].isna(), "relation_key"].astype(str).unique().tolist())
        raise RuntimeError(f"Missing relation_group for relation_keys: {bad[:20]}{'...' if len(bad)>20 else ''}")

    # Keep only finite x/y
    df["cos_improvement"] = df["cos_improvement"].astype(float)
    df[args.ycol] = df[args.ycol].astype(float)
    df = df[np.isfinite(df["cos_improvement"]) & np.isfinite(df[args.ycol])].copy()

    # Print per-model relation counts before intersection
    print("[info] after filter+finite: rows =", len(df))
    for mk in MODEL_ORDER:
        n_rel = df[df["model_key"].astype(str) == mk]["relation_key"].nunique()
        print(f"[info] {mk}: n_rel={n_rel}")

    # Enforce intersection across models (after filtering & finite x/y)
    inter_sorted: List[str] = []
    if args.enforce_intersection:
        rel_sets = []
        for mk in MODEL_ORDER:
            s = set(df[df["model_key"].astype(str) == mk]["relation_key"].astype(str).tolist())
            rel_sets.append(s)
        inter = set.intersection(*rel_sets) if rel_sets else set()
        inter_sorted = sorted(inter)
        df = df[df["relation_key"].astype(str).isin(inter_sorted)].copy()

        # Save intersection list
        suf = f"_{args.tag}" if args.tag else ""
        inter_path = os.path.join(args.outdir, f"intersection_relations{suf}.txt")
        with open(inter_path, "w", encoding="utf-8") as f:
            for r in inter_sorted:
                f.write(r + "\n")
        print(f"[info] intersection relations = {len(inter_sorted)}")
        print(f"[done] wrote: {inter_path}")

    # Save points used
    suf = f"_{args.tag}" if args.tag else ""
    points_path = os.path.join(args.outdir, f"points_used{suf}.csv")
    keep_cols = [c for c in ["model_key","relation_key","relation_group","n_test","cos_improvement",args.ycol,
                             "HALLUCINATION","REFUSAL","CORRECT","answered_n","noncorrect_n"]
                 if c in df.columns]
    df[keep_cols].to_csv(points_path, index=False)
    print(f"[done] wrote: {points_path}")

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
    x_all = df["cos_improvement"].to_numpy(float)
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    pad = 0.04 * (xmax - xmin) if xmax > xmin else 0.04
    xmin -= pad
    xmax += pad
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_key"].astype(str) == mk].copy()
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

        r_fmt = f"{r:.{args.title_r_decimals}f}" if np.isfinite(r) else "nan"
        if np.isfinite(p):
            p_fmt = f"{p:.{args.title_p_decimals}f}"
            title = f"{MODEL_TITLE.get(mk, mk)} (r={r_fmt}, p={p_fmt})"
        else:
            title = f"{MODEL_TITLE.get(mk, mk)} (r={r_fmt})"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    # Global labels (Figure2 style)
    fig.text(0.5, 0.05, "Δcos (mean over value-providing triples)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, args.ylabel, rotation=90, ha="center", va="center", fontsize=14)

    # Legend inside bottom-right panel
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

    tag = f"_{args.tag}" if args.tag else ""
    base = os.path.join(args.outdir, f"scatter_panel_4models_{args.ycol}{tag}_fit_fig2style")
    fig.savefig(base + ".png", dpi=args.dpi)
    fig.savefig(base + ".pdf")
    plt.close(fig)

    print("[done] wrote:", base + ".pdf")
    print("[done] wrote:", base + ".png")
    if pearsonr is None:
        print("[WARN] SciPy not found -> p-values are NaN. Install via: pip install scipy")


if __name__ == "__main__":
    main()
