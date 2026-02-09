#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Figure-2-style 2x2 scatter panels from LRE 3-way judge aggregation.

Outputs 4 dashboards:
  1) all47: Hall / (Hall + Correct)
  2) all47: Hall / (Hall + Refusal)
  3) min10: Hall / (Hall + Correct)   (after min-n_test filter; intersection across models)
  4) min10: Hall / (Hall + Refusal)  (after min-n_test filter; intersection across models)

Notes:
- Uses intersection relations across the 4 models for each dashboard.
- Fits a linear line (np.polyfit degree=1) to match the style of your existing Fig2 script.
- Computes Pearson r and (if SciPy is available) the standard two-sided p-value.
- Also writes Monte-Carlo permutation two-sided p-values to CSV (not printed in titles).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_count_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Infer (hall_col, ref_col, correct_col) from common column name variants."""
    hall = _first_existing(
        df,
        ["hall", "n_hall", "n_hallucination", "n_hallucinations", "hallucination", "hallucinations"],
    )
    ref = _first_existing(
        df,
        ["ref", "n_ref", "n_refusal", "n_refusals", "refusal", "refusals"],
    )
    corr = _first_existing(
        df,
        ["correct", "n_correct", "n_corrects", "n_cor", "cor", "n_right", "right"],
    )
    return hall, ref, corr


def _pearson_r_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
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


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _perm_test_pearson_two_sided(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int = 0) -> float:
    """Monte Carlo permutation test (two-sided) for Pearson r."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 3 or n_perm <= 0:
        return float("nan")

    r_obs = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r_obs):
        return float("nan")

    rng = np.random.default_rng(seed)
    count = 0
    abs_r_obs = abs(r_obs)

    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r_perm = float(np.corrcoef(x, y_perm)[0, 1])
        if abs(r_perm) >= abs_r_obs - 1e-12:
            count += 1

    # +1 smoothing (conservative)
    return (count + 1) / (n_perm + 1)


def _group_styles(groups: List[str]) -> Dict[str, Dict[str, str]]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    ordered = [g for g in GROUP_ORDER if g in groups] + [g for g in sorted(groups) if g not in GROUP_ORDER]
    out: Dict[str, Dict[str, str]] = {}
    for i, g in enumerate(ordered):
        out[g] = {"color": colors[i % len(colors)], "marker": markers[i % len(markers)]}
    return out


def _filter_intersection(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Keep only relations present (finite x/y) for all models; return filtered df + intersection list."""
    rel_sets = []
    for mk in MODEL_ORDER:
        sub = df[df["model_key"] == mk].copy()
        sub = sub[np.isfinite(sub[x_col].astype(float)) & np.isfinite(sub[y_col].astype(float))]
        rel_sets.append(set(sub["relation_key"].astype(str).tolist()))
    inter = set.intersection(*rel_sets) if rel_sets else set()
    inter_sorted = sorted(inter)
    out = df[df["relation_key"].astype(str).isin(inter_sorted)].copy()
    return out, inter_sorted


@dataclass
class MetricSpec:
    key: str
    y_label: str
    y_col: str


def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hall_over_value and hall_over_hallref from count columns (or reuse precomputed rate columns)."""
    df = df.copy()

    need = {"model_key", "relation_key", "cos_improvement"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Input CSV missing required columns: {sorted(miss)}")

    hall_col, ref_col, corr_col = _infer_count_cols(df)

    # If counts are missing but rates exist, reuse them.
    if hall_col is None or ref_col is None or corr_col is None:
        have_val = _first_existing(df, ["hall_over_value", "hall_rate_given_value", "hall_rate_value"])
        have_hr = _first_existing(df, ["hall_over_hallref", "hall_rate_over_refusal", "hall_rate_hall_ref"])
        if have_val is None or have_hr is None:
            raise ValueError(
                "Could not infer hall/ref/correct count columns, and could not find "
                "precomputed rate columns. Please ensure the CSV has counts "
                "(hall/ref/correct) or rate columns."
            )
        df["hall_over_value"] = df[have_val].astype(float)
        df["hall_over_hallref"] = df[have_hr].astype(float)
        return df

    hall = df[hall_col].astype(float)
    ref = df[ref_col].astype(float)
    corr = df[corr_col].astype(float)

    denom_val = hall + corr
    denom_hr = hall + ref

    df["hall_over_value"] = np.where(denom_val > 0, hall / denom_val, np.nan)
    df["hall_over_hallref"] = np.where(denom_hr > 0, hall / denom_hr, np.nan)

    df["n_value"] = denom_val
    df["n_hallref"] = denom_hr

    return df


def _plot_panel(df: pd.DataFrame, outdir: str, metric: MetricSpec, tag: str, n_perm: int, seed: int, dpi: int) -> None:
    os.makedirs(outdir, exist_ok=True)

    x_col = "cos_improvement"
    y_col = metric.y_col

    # Intersection filter
    df_f, inter = _filter_intersection(df, x_col=x_col, y_col=y_col)

    # Save intersection + points
    with open(os.path.join(outdir, f"intersection_relations_{tag}_{metric.key}.txt"), "w", encoding="utf-8") as f:
        for r in inter:
            f.write(r + "\n")
    df_f.to_csv(os.path.join(outdir, f"points_{tag}_{metric.key}.csv"), index=False)

    # Global x range
    x_all = df_f[x_col].astype(float).to_numpy()
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    pad = 0.04 * (xmax - xmin) if xmax > xmin else 0.04
    xmin -= pad
    xmax += pad

    y_min, y_max = -0.05, 1.05

    # Group styling (Figure-2 style)
    use_group = "relation_group" in df_f.columns
    if use_group:
        groups = df_f["relation_group"].astype(str).unique().tolist()
        styles = _group_styles(groups)
        legend_handles = [
            Line2D(
                [0], [0],
                marker=styles[g]["marker"], color="w",
                label=GROUP_TITLE.get(g, g),
                markerfacecolor=styles[g]["color"],
                markeredgecolor="black",
                markersize=8, linewidth=0
            )
            for g in ([g for g in GROUP_ORDER if g in styles] + [g for g in sorted(styles) if g not in GROUP_ORDER])
        ]
    else:
        styles = {"all": {"color": "C0", "marker": "o"}}
        legend_handles = []

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    stats_rows = []

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df_f[df_f["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub[x_col].astype(float).to_numpy()
        y = sub[y_col].astype(float).to_numpy()

        r, p = _pearson_r_p(x, y)
        p_perm = _perm_test_pearson_two_sided(x, y, n_perm=n_perm, seed=seed) if n_perm > 0 else float("nan")
        slope, intercept = _fit_line(x, y)

        n_pts = int(np.isfinite(x).sum())

        stats_rows.append({
            "tag": tag,
            "metric": metric.key,
            "model_key": mk,
            "n_rel": n_pts,
            "pearson_r": r,
            "p_two_ttest": p,
            "p_two_perm_mc": p_perm,
        })

        if use_group:
            for g, st in styles.items():
                ss = sub[sub["relation_group"].astype(str) == g]
                if ss.empty:
                    continue
                ax.scatter(
                    ss[x_col].astype(float),
                    ss[y_col].astype(float),
                    s=75,
                    marker=st["marker"],
                    facecolor=st["color"],
                    edgecolor="black",
                    linewidth=0.75,
                    zorder=3,
                )
        else:
            ax.scatter(
                x, y,
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

        # Titles show r + (ttest) p; permutation p is written to CSV.
        if np.isfinite(p):
            title = f"{MODEL_TITLE.get(mk, mk)} (n={n_pts}, r={r:.3f}, p={p:.4f})"
        else:
            title = f"{MODEL_TITLE.get(mk, mk)} (n={n_pts}, r={r:.3f})"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.05, "Relational linearity (Δcos)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, metric.y_label, rotation=90, ha="center", va="center", fontsize=14)

    if legend_handles:
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

    base = os.path.join(outdir, f"scatter_panel_4models_{metric.key}_{tag}_fit")
    fig.savefig(base + ".png", dpi=dpi)
    fig.savefig(base + ".pdf")
    plt.close(fig)

    stats_path = os.path.join(outdir, f"corr_summary_{metric.key}_{tag}.csv")
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-all47", required=True)
    ap.add_argument("--csv-min10", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-perm", type=int, default=20000, help="Monte Carlo permutations for two-sided p. 0 to skip.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df_all = pd.read_csv(args.csv_all47)
    df_min = pd.read_csv(args.csv_min10)

    df_all = _compute_metrics(df_all)
    df_min = _compute_metrics(df_min)

    metrics = [
        MetricSpec(
            key="hall_over_value",
            y_col="hall_over_value",
            y_label="Hallucination rate (Hall / (Hall + Correct))",
        ),
        MetricSpec(
            key="hall_over_hallref",
            y_col="hall_over_hallref",
            y_label="Hallucination rate (Hall / (Hall + Refusal))",
        ),
    ]

    for m in metrics:
        _plot_panel(df_all, outdir=args.outdir, metric=m, tag="all47", n_perm=args.n_perm, seed=args.seed, dpi=args.dpi)
        _plot_panel(df_min, outdir=args.outdir, metric=m, tag="min10", n_perm=args.n_perm, seed=args.seed, dpi=args.dpi)

    if pearsonr is None:
        print("[WARN] SciPy not found; p-values in titles will be NaN. Install: pip install scipy")

    print("[done] wrote plots + stats to:", args.outdir)


if __name__ == "__main__":
    main()
