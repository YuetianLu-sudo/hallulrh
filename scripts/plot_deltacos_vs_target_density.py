#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Δcos vs target-density proxy: |unique target values| / |#triples|.

- Density is computed from LRE prompts JSONL (gold_object) per relation.
- Δcos is taken from an input CSV (per model_key × relation_key).

Outputs:
  - scatter_panel_4models_deltacos_vs_density_<tag>.{pdf,png}
  - deltacos_vs_density_points_<tag>.csv
  - deltacos_vs_density_corr_<tag>.csv
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any, Iterable, Optional, Tuple, List

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

# Common key variants
REL_KEYS = ["relation_key", "relation", "task", "property", "predicate", "rel"]
MODEL_KEYS = ["model_key", "model", "model_name", "checkpoint"]
DELTACOS_KEYS = ["cos_improvement", "delta_cos_mean_test", "delta_cos_mean_value", "delta_cos"]


def find_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def load_density_from_prompts(prompts_path: str) -> pd.DataFrame:
    n_triples: Dict[str, int] = defaultdict(int)
    uniq_vals: Dict[str, set] = defaultdict(set)
    rel2group: Dict[str, str] = {}

    # robust gold keys
    GOLD_KEYS = ["gold_object", "gold_answer", "object", "answer", "target", "label"]

    with open(prompts_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            rel = rec.get("relation_key") or rec.get("relation") or rec.get("task") or ""
            if not isinstance(rel, str) or not rel.strip():
                continue
            rel = rel.strip()

            grp = rec.get("relation_group")
            if isinstance(grp, str) and grp.strip() and rel not in rel2group:
                rel2group[rel] = grp.strip()

            gold = None
            for k in GOLD_KEYS:
                if k in rec and rec[k] is not None:
                    gold = rec[k]
                    break
            if gold is None:
                continue
            if not isinstance(gold, str):
                gold = str(gold)
            gold = gold.strip()
            if not gold:
                continue

            n_triples[rel] += 1
            uniq_vals[rel].add(gold)

    rows = []
    for rel, n in n_triples.items():
        u = len(uniq_vals[rel])
        rows.append(
            {
                "relation_key": rel,
                "n_triples": int(n),
                "n_unique_targets": int(u),
                "target_density": float(u / n) if n > 0 else float("nan"),
                "relation_group": rel2group.get(rel, ""),
            }
        )

    return pd.DataFrame(rows)


def format_p(p: float) -> str:
    """Format p-values for plot titles (avoid showing 0.0000)."""
    import numpy as np
    if not np.isfinite(p):
        return "p=nan"
    # scipy may underflow extremely small p-values to 0.0
    if p == 0.0:
        return "p<1e-300"
    # Use scientific notation for tiny p; fixed decimals otherwise
    if p < 1e-4:
        return f"p={p:.2e}"
    return f"p={p:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV containing per-(model,relation) Δcos.")
    ap.add_argument("--prompts", required=True, help="LRE prompts JSONL with gold_object for density computation.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="run", help="Tag used in output filenames.")
    ap.add_argument("--deltacos-col", default=None, help="Override Δcos column name (default: auto-detect).")
    ap.add_argument("--intersection", action="store_true",
                    help="Keep only relations that appear for ALL 4 models in MODEL_ORDER.")
    ap.add_argument("--min-n-triples", type=int, default=1,
                    help="Filter relations with too few triples in prompts (default: 1).")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    rel_col = find_first_col(df, REL_KEYS)
    model_col = find_first_col(df, MODEL_KEYS)
    if rel_col is None or model_col is None:
        raise SystemExit(f"[error] Could not find relation/model columns. "
                         f"Have cols={list(df.columns)}")

    if args.deltacos_col is not None:
        y_col = args.deltacos_col
        if y_col not in df.columns:
            raise SystemExit(f"[error] --deltacos-col {y_col} not found in CSV columns.")
    else:
        y_col = find_first_col(df, DELTACOS_KEYS)
        if y_col is None:
            raise SystemExit(f"[error] Could not auto-detect Δcos column in CSV. "
                             f"Tried {DELTACOS_KEYS}. Have cols={list(df.columns)}")

    # normalize columns
    df = df.rename(columns={rel_col: "relation_key", model_col: "model_key", y_col: "cos_improvement"}).copy()
    df["relation_key"] = df["relation_key"].astype(str)
    df["model_key"] = df["model_key"].astype(str)

    # If duplicates exist, average them (safe guard)
    df = df.groupby(["model_key", "relation_key"], as_index=False).agg(
        cos_improvement=("cos_improvement", "mean"),
        **({} if "relation_group" not in df.columns else {"relation_group": ("relation_group", "first")}),
        **({} if "n_test" not in df.columns else {"n_test": ("n_test", "first")}),
    )

    # load density table (relation-level, model-independent)
    dens = load_density_from_prompts(args.prompts)
    if args.min_n_triples > 1:
        dens = dens[dens["n_triples"] >= args.min_n_triples].copy()

    # merge
    m = df.merge(dens[["relation_key", "n_triples", "n_unique_targets", "target_density", "relation_group"]],
                 on="relation_key", how="left", suffixes=("", "_from_prompts"))

    # fill relation_group if missing
    if "relation_group_x" in m.columns and "relation_group_y" in m.columns:
        # rare case due to suffixing; normalize
        m["relation_group"] = m["relation_group_x"].fillna(m["relation_group_y"])
        m = m.drop(columns=["relation_group_x", "relation_group_y"], errors="ignore")
    elif "relation_group" not in m.columns:
        m["relation_group"] = ""

    missing = int(m["target_density"].isna().sum())
    if missing > 0:
        bad = sorted(m.loc[m["target_density"].isna(), "relation_key"].unique().tolist())
        raise SystemExit(f"[error] Missing density for {missing} rows. "
                         f"Example missing relation_keys={bad[:20]}")

    # optional: intersection relations across 4 models
    use_models = [mk for mk in MODEL_ORDER if mk in set(m["model_key"].unique())]
    if args.intersection:
        rel_sets = [set(m[m["model_key"] == mk]["relation_key"].tolist()) for mk in use_models]
        inter = set.intersection(*rel_sets) if rel_sets else set()
        m = m[m["relation_key"].isin(sorted(inter))].copy()
        print(f"[filter] intersection relations={len(inter)} across models={use_models}")

    # write points used
    points_path = os.path.join(args.outdir, f"deltacos_vs_density_points_{args.tag}.csv")
    m.to_csv(points_path, index=False)
    print("[done] wrote:", points_path)

    # correlation summary
    rows = []
    for mk in use_models:
        sub = m[m["model_key"] == mk]
        r, p = corr_r_p(sub["target_density"].to_numpy(float), sub["cos_improvement"].to_numpy(float))
        rows.append({"model_key": mk, "n_rel": int(len(sub)), "r": r, "p_two": p})

    # pooled FE (demean within model)
    g = m["model_key"].astype(str)
    x_res = m["target_density"] - m.groupby(g)["target_density"].transform("mean")
    y_res = m["cos_improvement"] - m.groupby(g)["cos_improvement"].transform("mean")
    r_pool, p_pool = corr_r_p(x_res.to_numpy(float), y_res.to_numpy(float))
    rows.append({"model_key": "POOLED_FE", "n_rel": int(len(m)), "r": r_pool, "p_two": p_pool})

    corr_path = os.path.join(args.outdir, f"deltacos_vs_density_corr_{args.tag}.csv")
    pd.DataFrame(rows).to_csv(corr_path, index=False)
    print("[done] wrote:", corr_path)

    # styles (match Figure2 palette order: C0,C1,C2,C3)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4"]
    styles = {g: {"color": colors[i % len(colors)], "marker": MARKER_MAP.get(g, "o")}
              for i, g in enumerate(GROUP_ORDER)}

    legend_handles = [
        Line2D([0], [0],
               marker=styles[g]["marker"], color="w",
               label=GROUP_TITLE.get(g, g),
               markerfacecolor=styles[g]["color"],
               markeredgecolor="black",
               markersize=8, linewidth=0)
        for g in GROUP_ORDER
    ]

    # global axis range
    x_all = m["target_density"].to_numpy(float)
    y_all = m["cos_improvement"].to_numpy(float)

    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))

    xpad = 0.06 * (xmax - xmin) if xmax > xmin else 0.02
    ypad = 0.06 * (ymax - ymin) if ymax > ymin else 0.02

    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = m[m["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub["target_density"].to_numpy(float)
        y = sub["cos_improvement"].to_numpy(float)
        r, p = corr_r_p(x, y)
        slope, intercept = fit_line(x, y)

        # scatter by group (if group missing, everything falls into "factual" style visually)
        for g in GROUP_ORDER:
            ss = sub[sub["relation_group"].astype(str) == g]
            if ss.empty:
                continue
            ax.scatter(
                ss["target_density"], ss["cos_improvement"],
                s=75,
                marker=styles[g]["marker"],
                facecolor=styles[g]["color"],
                edgecolor="black",
                linewidth=0.75,
                zorder=3,
            )

        # if some groups missing, plot remaining rows (unknown groups) in default marker
        unk = sub[~sub["relation_group"].astype(str).isin(GROUP_ORDER)]
        if not unk.empty:
            ax.scatter(
                unk["target_density"], unk["cos_improvement"],
                s=75,
                marker="o",
                facecolor=colors[0],
                edgecolor="black",
                linewidth=0.75,
                zorder=3,
            )

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(xmin, xmax, 200)
            ax.plot(xx, slope * xx + intercept, linestyle="--", linewidth=1.6,
                    color="black", alpha=0.55, zorder=2)

        title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f}, {format_p(p)})" if np.isfinite(p) else f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f})"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.05, r"Output density: $|V|/|T|$ (unique targets / triples)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, r"$\Delta$cos (mean on TEST triples)", rotation=90, ha="center", va="center", fontsize=14)

    # Legend inside bottom-right panel
    leg = axes[3].legend(
        handles=legend_handles,
        title="Relation group",
        loc="upper right",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        borderaxespad=0.4,
        labelspacing=0.35,
        handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.28)

    base = os.path.join(args.outdir, f"scatter_panel_4models_deltacos_vs_density_{args.tag}")
    fig.savefig(base + ".png", dpi=args.dpi)
    fig.savefig(base + ".pdf")
    plt.close(fig)

    print("[done] wrote:", base + ".pdf")
    print("[done] wrote:", base + ".png")

    if pearsonr is None:
        print("[WARN] SciPy not found -> p-values are NaN. Install via: pip install scipy")


if __name__ == "__main__":
    main()
