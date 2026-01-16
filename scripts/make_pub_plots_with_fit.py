#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Tuple, List, Optional
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def pearson_r_p_two(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Return (r, p_two) for Pearson correlation.
    p_two is the standard two-sided t-test p-value (df = n-2), via SciPy if available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size < 3:
        return float("nan"), float("nan")

    try:
        import scipy.stats as st  # type: ignore
        r, p = st.pearsonr(x, y)  # two-sided p-value
        return float(r), float(p)
    except Exception:
        # Fallback: r only; p unavailable without SciPy
        return pearson_r(x, y), float("nan")


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan"), float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def exact_perm_p_pearson_one_two(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, int]:
    """
    Exact permutation test for Pearson correlation (n small).
    One-sided alternative: r_perm >= r_obs.
    Two-sided: |r_perm| >= |r_obs|.
    Enumerates all n! permutations of y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    n = int(x.size)
    r_obs = pearson_r(x, y)
    if not np.isfinite(r_obs) or n < 2:
        return float("nan"), float("nan"), 0

    idx = list(range(n))
    ge = 0
    abs_ge = 0
    n_perm = 0

    for perm in permutations(idx):
        y_perm = y[list(perm)]
        r_perm = pearson_r(x, y_perm)
        if not np.isfinite(r_perm):
            continue
        n_perm += 1
        if r_perm >= r_obs - eps:
            ge += 1
        if abs(r_perm) >= abs(r_obs) - eps:
            abs_ge += 1

    if n_perm == 0:
        return float("nan"), float("nan"), 0

    return ge / n_perm, abs_ge / n_perm, n_perm


def _relation_style_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    color_map = {rel: colors[i % len(colors)] for i, rel in enumerate(REL_ORDER)}
    marker_list = ["o", "s", "^", "D", "P", "X"]
    marker_map = {rel: marker_list[i % len(marker_list)] for i, rel in enumerate(REL_ORDER)}
    return color_map, marker_map


def _model_color_map() -> Dict[str, str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3"]
    return {mk: colors[i % len(colors)] for i, mk in enumerate(MODEL_ORDER)}


def save_figure(fig: plt.Figure, base_path_no_ext: str, dpi: int = 300, bbox_inches=None) -> None:
    png_path = base_path_no_ext + ".png"
    pdf_path = base_path_no_ext + ".pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(pdf_path, bbox_inches=bbox_inches)


def plot_behavior_bars(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """Per-model horizontal stacked bars (existing behavior)."""
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
        ax.set_ylim(len(rels) - 0.5, -0.5)  # top-to-bottom ordering

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


def plot_behavior_panel_4models(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """
    4-model dashboard for behavior bars.
    Fixes requested:
      - show relation names on y-axis (left column),
      - add a legend somewhere (global legend),
      - keep compact dashboard layout.
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for ax_i, (ax, mk) in enumerate(zip(axes, MODEL_ORDER)):
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
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

        ax.barh(
            y,
            halluc,
            xerr=halluc_err,
            capsize=2.5,
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
            capsize=2.5,
            label="Refusal",
            alpha=0.60,
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
        )

        ax.set_title(MODEL_TITLE.get(mk, mk), fontsize=13)
        ax.set_xlim(0.0, 1.0)

        ax.set_yticks(y)
        # ✅ Fix (2): show relation names on the y-axis (left column only, to save space)
        if ax_i % 2 == 0:
            ax.set_yticklabels(rels, fontsize=11)
        else:
            ax.set_yticklabels([])

        ax.set_ylim(len(rels) - 0.5, -0.5)  # consistent top-to-bottom ordering

        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

        # Grab legend handles/labels once (from the first plotted axis).
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # ✅ Fix (3): add a legend to the dashboard (global, compact, non-overlapping)
    if legend_handles and legend_labels:
        leg = fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=True,
            fontsize=11,
        )
        leg.get_frame().set_linewidth(0.8)

    fig.text(0.5, 0.075, "Rate", ha="center", va="center", fontsize=14)

    fig.subplots_adjust(left=0.18, right=0.99, top=0.92, bottom=0.12, wspace=0.18, hspace=0.28)

    base = os.path.join(outdir, "behavior_panel_4models_pub")
    save_figure(fig, base, dpi=dpi)
    plt.close(fig)


def plot_scatter_per_model_with_fit(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """Per-model scatter plots (existing behavior)."""
    os.makedirs(outdir, exist_ok=True)
    color_map, marker_map = _relation_style_maps()

    for mk in MODEL_ORDER:
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            continue
        sub = sub.set_index("relation").reindex(REL_ORDER).reset_index()

        x = sub["cos_improvement"].to_numpy(dtype=float)
        y = sub["halluc_rate"].to_numpy(dtype=float)

        slope, intercept = fit_line(x, y)

        fig, ax = plt.subplots(figsize=(5.4, 4.1))

        for rel in REL_ORDER:
            row = sub[sub["relation"] == rel]
            if row.empty:
                continue
            ax.scatter(
                float(row["cos_improvement"].iat[0]),
                float(row["halluc_rate"].iat[0]),
                s=120,
                marker=marker_map[rel],
                facecolor=color_map[rel],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                clip_on=False,
            )

        xmin = float(np.nanmin(x)) - 0.04
        xmax = float(np.nanmax(x)) + 0.04
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6))

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(xmin, xmax, 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, linestyle="--", linewidth=1.8, color="black", alpha=0.6, zorder=2)

        r, p_two = pearson_r_p_two(x, y)
        if np.isfinite(p_two):
            ax.set_title(f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f}, p={p_two:.4f})", fontsize=13)
        else:
            ax.set_title(f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f})", fontsize=13)

        ax.set_xlabel("LRE cosine improvement (Δcos)", fontsize=12)
        ax.set_ylabel("Hallucination rate", fontsize=12)

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
        base = os.path.join(outdir, f"{mk}_lre_vs_halluc_scatter_fit")
        save_figure(fig, base, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_scatter_panel_4models_with_fit(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """
    4-model dashboard scatter with fit lines.
    We show per-model Pearson r and the *two-sided t-test* p-value (standard).
    Exact permutation p-values remain in Appendix/Table outputs (not in the main figure).
    """
    os.makedirs(outdir, exist_ok=True)
    color_map, marker_map = _relation_style_maps()

    global_xmin = float(df["cos_improvement"].min()) - 0.04
    global_xmax = float(df["cos_improvement"].max()) + 0.04
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    # Build relation legend handles once.
    rel_handles = [
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

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.set_index("relation").reindex(REL_ORDER).reset_index()

        x = sub["cos_improvement"].to_numpy(dtype=float)
        y = sub["halluc_rate"].to_numpy(dtype=float)

        r, p_two = pearson_r_p_two(x, y)
        slope, intercept = fit_line(x, y)

        for rel in REL_ORDER:
            row = sub[sub["relation"] == rel]
            if row.empty:
                continue
            ax.scatter(
                float(row["cos_improvement"].iat[0]),
                float(row["halluc_rate"].iat[0]),
                s=110,
                marker=marker_map[rel],
                facecolor=color_map[rel],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                clip_on=False,
            )

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(global_xmin, global_xmax, 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, linestyle="--", linewidth=1.6, color="black", alpha=0.55, zorder=2)

        # Title: r (3 decimals) + p (two-sided t-test, 4 decimals)
        if np.isfinite(p_two):
            title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f}, p={p_two:.4f})"
        else:
            title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f})"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))

        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    # Global axis labels
    fig.text(0.5, 0.05, "LRE cosine improvement (Δcos)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, "Hallucination rate", rotation=90, ha="center", va="center", fontsize=14)

    # Put legend INSIDE Qwen subplot (bottom-right)
    qwen_ax = axes[3]
    q_leg = qwen_ax.legend(
        handles=rel_handles,
        title="Relation",
        loc="lower right",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        borderaxespad=0.4,
        labelspacing=0.4,
        handletextpad=0.5,
    )
    q_leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.28)

    base = os.path.join(outdir, "scatter_panel_4models_fit")
    save_figure(fig, base, dpi=dpi)
    plt.close(fig)


def plot_scatter_all_24_with_fit(df: pd.DataFrame, outdir: str, dpi: int = 300) -> None:
    """All 24 points scatter (existing behavior)."""
    os.makedirs(outdir, exist_ok=True)

    model_color = _model_color_map()
    _, marker_map = _relation_style_maps()

    sub = df[df["model_key"].isin(MODEL_ORDER) & df["relation"].isin(REL_ORDER)].copy()
    if sub.empty:
        return

    x = sub["cos_improvement"].to_numpy(dtype=float)
    y = sub["halluc_rate"].to_numpy(dtype=float)
    r = pearson_r(x, y)
    slope, intercept = fit_line(x, y)

    xmin = float(np.nanmin(x)) - 0.04
    xmax = float(np.nanmax(x)) + 0.04

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    for _, row in sub.iterrows():
        mk = str(row["model_key"])
        rel = str(row["relation"])
        ax.scatter(
            float(row["cos_improvement"]),
            float(row["halluc_rate"]),
            s=110,
            marker=marker_map[rel],
            facecolor=model_color.get(mk, "C0"),
            edgecolor="black",
            linewidth=0.75,
            zorder=3,
            clip_on=False,
        )

    if np.isfinite(slope) and np.isfinite(intercept):
        xx = np.linspace(xmin, xmax, 200)
        yy = slope * xx + intercept
        ax.plot(xx, yy, linestyle="--", linewidth=1.8, color="black", alpha=0.6, zorder=2)

    ax.set_title(f"All models (n=24) (r={r:.2f})", fontsize=13)
    ax.set_xlabel("LRE cosine improvement (Δcos)", fontsize=12)
    ax.set_ylabel("Hallucination rate", fontsize=12)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    model_handles: List[Line2D] = []
    for mk in MODEL_ORDER:
        model_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=MODEL_TITLE.get(mk, mk),
                markerfacecolor=model_color.get(mk, "C0"),
                markeredgecolor="black",
                markersize=9,
                linewidth=0,
            )
        )

    rel_handles: List[Line2D] = []
    for rel in REL_ORDER:
        rel_handles.append(
            Line2D(
                [0], [0],
                marker=marker_map[rel],
                color="black",
                label=REL_LABEL.get(rel, rel),
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=9,
                linewidth=0,
            )
        )

    leg1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.02, 0.70),
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=rel_handles,
        title="Relation",
        loc="center left",
        bbox_to_anchor=(1.02, 0.25),
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )

    fig.tight_layout()
    base = os.path.join(outdir, "scatter_all_24_fit")
    save_figure(fig, base, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to behavior_plus_lre.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG outputs (default: 300)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {"model_key", "relation", "halluc_rate", "refusal_rate", "cos_improvement"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Existing per-model plots
    plot_behavior_bars(df, outdir=outdir, dpi=args.dpi)
    plot_scatter_per_model_with_fit(df, outdir=outdir, dpi=args.dpi)

    # Dashboards
    plot_behavior_panel_4models(df, outdir=outdir, dpi=args.dpi)
    plot_scatter_panel_4models_with_fit(df, outdir=outdir, dpi=args.dpi)

    # Optional pooled scatter
    plot_scatter_all_24_with_fit(df, outdir=outdir, dpi=args.dpi)

    print(f"[plots] Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()