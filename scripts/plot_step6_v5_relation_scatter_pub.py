#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pub-ready Step6(v5) scatter + p-values (Figure-1 style), using the *intersection* relations.

Default filter: n_value > 10  (implemented as n_value >= 11).

Input can be:
  - step6 output directory containing relation_summary.csv.gz
  - a direct path to relation_summary.csv.gz
  - the step6 .zip archive (containing */relation_summary.csv.gz)

Outputs to --outdir:
  - scatter_panel_4models_acc_given_value_fit.{pdf,png}
  - (optional) scatter_panel_4models_hall_rate_given_value_fit.{pdf,png}
  - corr_summary_acc_given_value.csv  (per-model + pooled FE)
  - points_filtered_intersection.csv
  - intersection_relations.txt
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
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


def read_relation_summary(inp: str) -> pd.DataFrame:
    if os.path.isdir(inp):
        p = os.path.join(inp, "relation_summary.csv.gz")
        return pd.read_csv(p, compression="gzip")

    if inp.endswith(".zip"):
        with zipfile.ZipFile(inp, "r") as z:
            members = [n for n in z.namelist() if n.endswith("relation_summary.csv.gz")]
            if not members:
                raise FileNotFoundError(f"No relation_summary.csv.gz found inside zip: {inp}")
            member = sorted(members, key=len)[0]
            raw = z.read(member)
        return pd.read_csv(io.BytesIO(raw), compression="gzip")

    if inp.endswith(".csv.gz"):
        return pd.read_csv(inp, compression="gzip")

    return pd.read_csv(inp)


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


def filter_intersection(df: pd.DataFrame, models: List[str], min_n_value: int) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["model_key"] = df["model_key"].astype(str)
    df["relation_key"] = df["relation_key"].astype(str)
    df = df[df["model_key"].isin(models)]
    df = df[df["n_value"].astype(float) >= float(min_n_value)]

    rel_sets = [set(df[df["model_key"] == mk]["relation_key"].tolist()) for mk in models]
    inter = set.intersection(*rel_sets) if rel_sets else set()
    inter_sorted = sorted(inter)
    df = df[df["relation_key"].isin(inter_sorted)].copy()
    return df, inter_sorted


def group_styles(groups: List[str]) -> Dict[str, Dict[str, str]]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    ordered = [g for g in GROUP_ORDER if g in groups] + [g for g in sorted(groups) if g not in GROUP_ORDER]
    out = {}
    for i, g in enumerate(ordered):
        out[g] = {"color": colors[i % len(colors)], "marker": markers[i % len(markers)]}
    return out


def plot_panel(df: pd.DataFrame, outdir: str, y_col: str, y_label: str, dpi: int = 300) -> str:
    x_col = "delta_cos_mean_value"

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
            Line2D([0], [0],
                   marker=styles[g]["marker"], color="w",
                   label=GROUP_TITLE.get(g, g),
                   markerfacecolor=styles[g]["color"],
                   markeredgecolor="black",
                   markersize=8, linewidth=0)
            for g in ([g for g in GROUP_ORDER if g in styles] + [g for g in sorted(styles) if g not in GROUP_ORDER])
        ]
    else:
        styles = {"all": {"color": "C0", "marker": "o"}}
        legend_handles = []

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_key"] == mk]
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
                ax.scatter(ss[x_col], ss[y_col], s=75, marker=st["marker"],
                           facecolor=st["color"], edgecolor="black", linewidth=0.75, zorder=3)
        else:
            ax.scatter(x, y, s=75, marker=styles["all"]["marker"],
                       facecolor=styles["all"]["color"], edgecolor="black", linewidth=0.75, zorder=3)

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(xmin, xmax, 200)
            ax.plot(xx, slope * xx + intercept, linestyle="--", linewidth=1.6, color="black", alpha=0.55, zorder=2)

        title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f}, p={p:.4f})" if np.isfinite(p) else f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f})"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.05, "Δcos (mean over value-providing triples)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, y_label, rotation=90, ha="center", va="center", fontsize=14)

    if legend_handles:
        leg = axes[3].legend(handles=legend_handles, title="Relation group", loc="lower right",
                             frameon=True, fontsize=10, title_fontsize=11,
                             borderaxespad=0.4, labelspacing=0.35, handletextpad=0.5)
        leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.28)

    base = os.path.join(outdir, f"scatter_panel_4models_{y_col}_fit")
    fig.savefig(base + ".png", dpi=dpi)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    return base + ".pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="step6 outdir OR relation_summary.csv.gz OR step6 zip")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_n_value", type=int, default=11, help="default 11 (paper wording: n_value>10)")
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--make_both", action="store_true", help="also output Hall|Value plot + stats")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    df = read_relation_summary(args.input)

    need = {"model_key", "relation_key", "n_value", "delta_cos_mean_value"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"relation_summary missing required columns: {sorted(miss)}")

    # ensure both metrics exist
    if "acc_given_value" not in df.columns:
        if "hall_rate_given_value" in df.columns:
            df["acc_given_value"] = 1.0 - df["hall_rate_given_value"].astype(float)
        else:
            raise ValueError("Need acc_given_value or hall_rate_given_value in relation_summary.")
    if "hall_rate_given_value" not in df.columns:
        df["hall_rate_given_value"] = 1.0 - df["acc_given_value"].astype(float)

    df_f, inter = filter_intersection(df, models=models, min_n_value=args.min_n_value)
    if df_f.empty:
        raise RuntimeError("No data left after filtering; check input/min_n_value/models.")

    # save intersection list
    with open(os.path.join(args.outdir, "intersection_relations.txt"), "w", encoding="utf-8") as f:
        for r in inter:
            f.write(r + "\n")

    # save plotted points
    keep_cols = [c for c in ["model_key", "relation_key", "relation_group", "relation_name",
                             "n_value", "delta_cos_mean_value", "acc_given_value", "hall_rate_given_value"]
                 if c in df_f.columns]
    df_f[keep_cols].to_csv(os.path.join(args.outdir, "points_filtered_intersection.csv"), index=False)

    print(f"[filter] min_n_value={args.min_n_value} (paper wording: n_value>{args.min_n_value-1})")
    print(f"[filter] intersection relations={len(inter)}")

    # stats: per-model + pooled FE
    def write_stats(metric_col: str, out_csv: str) -> None:
        rows = []
        for mk in MODEL_ORDER:
            if mk not in models:
                continue
            sub = df_f[df_f["model_key"] == mk]
            r, p = corr_r_p(sub["delta_cos_mean_value"], sub[metric_col])
            rows.append({"metric": metric_col, "model_key": mk, "n_rel": int(len(sub)), "r": r, "p_two": p})

        # pooled FE: demean within model
        g = df_f["model_key"].astype(str)
        x_res = df_f["delta_cos_mean_value"] - df_f.groupby(g)["delta_cos_mean_value"].transform("mean")
        y_res = df_f[metric_col] - df_f.groupby(g)[metric_col].transform("mean")
        r_pool, p_pool = corr_r_p(x_res.to_numpy(float), y_res.to_numpy(float))
        rows.append({"metric": metric_col, "model_key": "POOLED_FE", "n_rel": int(len(df_f)), "r": r_pool, "p_two": p_pool})

        pd.DataFrame(rows).to_csv(out_csv, index=False)

        print(f"\n== Pearson (two-sided p): {metric_col} vs delta_cos_mean_value ==")
        for row in rows:
            if row["model_key"] == "POOLED_FE":
                print(f"{row['model_key']}: n_points={row['n_rel']} r={row['r']:.6f} p={row['p_two']:.6g}")
            else:
                print(f"{row['model_key']}: n_rel={row['n_rel']} r={row['r']:.6f} p={row['p_two']:.6g}")
        print(f"[stats] wrote {out_csv}")

    # Acc|Value (recommended)
    write_stats("acc_given_value", os.path.join(args.outdir, "corr_summary_acc_given_value.csv"))
    pdf1 = plot_panel(df_f, args.outdir, "acc_given_value", "Accuracy | Value (Acc|Value)", dpi=args.dpi)
    print(f"[plot] wrote {pdf1}")

    if args.make_both:
        write_stats("hall_rate_given_value", os.path.join(args.outdir, "corr_summary_hall_rate_given_value.csv"))
        pdf2 = plot_panel(df_f, args.outdir, "hall_rate_given_value", "Hallucination | Value (Hall|Value)", dpi=args.dpi)
        print(f"[plot] wrote {pdf2}")

    if pearsonr is None:
        print("\n[WARN] SciPy not found, so p-values are NaN. Install it via: pip install scipy")


if __name__ == "__main__":
    main()
