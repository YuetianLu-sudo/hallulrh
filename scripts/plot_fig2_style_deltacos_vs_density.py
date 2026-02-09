#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure2-style dashboard (2x2) for:
  x = Δcos (cos_improvement; mean over TEST triples)
  y = |unique target values| / |triples|  (computed from gold_object in prompts JSONL)

Key options:
  --min-n-test: filter relations by n_test >= min_n_test (use 11 to match "n_test > 10")
  --intersection: keep only relations that survive the filter in ALL 4 models (Fig2-style)
Outputs:
  - points_used_*.csv
  - corr_summary_*.csv
  - scatter_panel_4models_deltacos_vs_density_*.{png,pdf}
"""

import argparse
import json
import os
from collections import defaultdict
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

MARKER_MAP = {
    "factual": "o",
    "commonsense": "s",
    "linguistic": "^",
    "bias": "D",
}

GOLD_KEYS = ["gold_object", "object", "gold_answer", "target", "label", "y", "obj"]


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
    r, p = pearsonr(x, y)  # two-sided
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


def _pick_gold(rec: Dict) -> str:
    for k in GOLD_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def compute_density(prompts_path: str, rels_needed: set) -> pd.DataFrame:
    n_triples = defaultdict(int)
    uniq = defaultdict(set)
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
            if rk is None:
                continue
            rk = str(rk)
            if rk not in rels_needed:
                continue

            n_triples[rk] += 1
            gold = _pick_gold(rec)
            if gold:
                uniq[rk].add(gold)

            grp = rec.get("relation_group")
            if rk not in rel2group and isinstance(grp, str) and grp.strip():
                rel2group[rk] = grp.strip()

    rows = []
    for rk in sorted(rels_needed):
        tot = int(n_triples.get(rk, 0))
        nu = int(len(uniq.get(rk, set())))
        dens = float(nu / tot) if tot > 0 else float("nan")
        rows.append(
            {
                "relation_key": rk,
                "n_triples": tot,
                "n_unique_targets": nu,
                "target_density": dens,
                "relation_group_from_prompts": rel2group.get(rk, ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="lre3way_behavior_plus_deltacos_all47.csv (or similar)")
    ap.add_argument("--prompts", required=True, help="lre_prompts_qonly.jsonl (must contain gold_object)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-n-test", type=int, default=11, help="Use 11 to match paper filter: n_test > 10")
    ap.add_argument("--intersection", action="store_true", help="Keep only relations present in all 4 models after filtering")
    ap.add_argument("--tag", default="fig2", help="Tag added to output filenames")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = {"model_key", "relation_key", "cos_improvement", "n_test"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"[error] CSV missing required columns: {sorted(miss)}")

    df["model_key"] = df["model_key"].astype(str)
    df["relation_key"] = df["relation_key"].astype(str)
    df = df[df["model_key"].isin(MODEL_ORDER)].copy()

    # Filter by n_test >= min_n_test (paper wording: n_test > 10 => min_n_test=11)
    df = df[df["n_test"].astype(float) >= float(args.min_n_test)].copy()

    # Intersection across models (Fig2 style)
    if args.intersection:
        rel_sets = []
        for mk in MODEL_ORDER:
            rel_sets.append(set(df[df["model_key"] == mk]["relation_key"].tolist()))
        inter = set.intersection(*rel_sets) if rel_sets else set()
        df = df[df["relation_key"].isin(sorted(inter))].copy()
    else:
        inter = set(df["relation_key"].unique().tolist())

    if not inter:
        raise SystemExit("[error] No relations left after filtering/intersection.")

    # Compute target density from prompts (gold targets)
    dens = compute_density(args.prompts, rels_needed=inter)

    # Merge density into df
    m = df.merge(dens, on="relation_key", how="left")

    if int(m["target_density"].isna().sum()) > 0:
        bad = sorted(m.loc[m["target_density"].isna(), "relation_key"].unique().tolist())
        raise SystemExit(f"[error] Missing density for relations: {bad[:20]}{'...' if len(bad) > 20 else ''}")

    # Prefer relation_group from df if exists, otherwise use prompts mapping
    if "relation_group" not in m.columns or m["relation_group"].isna().any():
        m["relation_group"] = m.get("relation_group", pd.Series([""] * len(m))).astype(str)
        fill = m["relation_group"].astype(str).str.strip() == ""
        m.loc[fill, "relation_group"] = m.loc[fill, "relation_group_from_prompts"]

    # Save points used
    suffix = f"min{args.min_n_test}" + ("_intersection" if args.intersection else "")
    points_path = os.path.join(args.outdir, f"points_deltacos_vs_density_{args.tag}_{suffix}.csv")
    m.to_csv(points_path, index=False)
    print("[done] wrote:", points_path)
    print("[info] unique relations per model:")
    print(m.groupby("model_key")["relation_key"].nunique())

    # Correlation summary per model
    rows = []
    for mk in MODEL_ORDER:
        sub = m[m["model_key"] == mk].copy()
        x = sub["cos_improvement"].to_numpy(float)
        y = sub["target_density"].to_numpy(float)
        r, p = corr_r_p(x, y)
        rows.append({"model_key": mk, "n_rel": int(len(sub)), "r": r, "p_two": p})
    corr_path = os.path.join(args.outdir, f"corr_deltacos_vs_density_{args.tag}_{suffix}.csv")
    pd.DataFrame(rows).to_csv(corr_path, index=False)
    print("[done] wrote:", corr_path)

    # -------- Plot (Figure2-style) --------
    # Fixed group styles (match Figure2: C0/C1/C2/C3)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4"]
    styles = {g: {"color": colors[i % len(colors)], "marker": MARKER_MAP[g]} for i, g in enumerate(GROUP_ORDER)}

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

    x_all = m["cos_improvement"].to_numpy(float)
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    pad = 0.04 * (xmax - xmin) if xmax > xmin else 0.04
    xmin -= pad
    xmax += pad

    # Density is within [0,1] by construction; keep Fig2 y-range style
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = m[m["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub["cos_improvement"].to_numpy(float)
        y = sub["target_density"].to_numpy(float)
        r, p = corr_r_p(x, y)
        slope, intercept = fit_line(x, y)

        # scatter by group
        for g in GROUP_ORDER:
            ss = sub[sub["relation_group"].astype(str) == g]
            if ss.empty:
                continue
            ax.scatter(
                ss["cos_improvement"], ss["target_density"],
                s=75,
                marker=styles[g]["marker"],
                facecolor=styles[g]["color"],
                edgecolor="black",
                linewidth=0.75,
                zorder=3,
            )

        # fit line
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

    fig.text(0.5, 0.05, "LRE cosine improvement (Δcos) — mean over TEST triples", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, "|unique targets| / |triples|  (target density)", rotation=90, ha="center", va="center", fontsize=14)

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

    base = os.path.join(args.outdir, f"scatter_panel_4models_deltacos_vs_density_{args.tag}_{suffix}_fig2style")
    fig.savefig(base + ".png", dpi=args.dpi)
    fig.savefig(base + ".pdf")
    plt.close(fig)

    print("[done] wrote:", base + ".png")
    print("[done] wrote:", base + ".pdf")

    if pearsonr is None:
        print("[WARN] SciPy not found -> p-values are NaN. Install via: pip install scipy")


if __name__ == "__main__":
    main()
