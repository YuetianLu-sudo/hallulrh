#!/usr/bin/env python3
import argparse, glob, os, re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ---------------- stats helpers ----------------

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def pearson_r_p_two(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        return float("nan"), float("nan")
    try:
        import scipy.stats as st  # type: ignore
        r, p = st.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return pearson_r(x, y), float("nan")

def spearman_r_p_two(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        return float("nan"), float("nan")
    try:
        import scipy.stats as st  # type: ignore
        rho, p = st.spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        return float("nan"), float("nan")

def weighted_pearson_r(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[m]; y = y[m]; w = w[m]
    if x.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan")

    w = w / np.sum(w)
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))

def approx_perm_p(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """
    Approx permutation test for Pearson r.
    Returns (p_one, p_two):
      p_one = P(r_perm >= r_obs)
      p_two = P(|r_perm| >= |r_obs|)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    r_obs = pearson_r(x, y)
    if not np.isfinite(r_obs) or n < 2:
        return float("nan"), float("nan")
    ge = 0
    abs_ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp = pearson_r(x, yp)
        if rp >= r_obs - 1e-12:
            ge += 1
        if abs(rp) >= abs(r_obs) - 1e-12:
            abs_ge += 1
    return ge / n_perm, abs_ge / n_perm

def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return float("nan"), float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)

def save_fig(fig: plt.Figure, base_no_ext: str, dpi: int = 300) -> None:
    fig.savefig(base_no_ext + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base_no_ext + ".pdf", bbox_inches="tight")

# ---------------- deltacos loading ----------------

def canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    want = {canon(c) for c in candidates}
    for col in df.columns:
        if canon(col) in want:
            return col
    return None

@dataclass
class DeltaCosTable:
    path: str
    df: pd.DataFrame
    model_col: str
    rel_col: str
    delta_col: str

def load_deltacos(path: str) -> DeltaCosTable:
    df = pd.read_csv(path)

    model_col = pick_col(df, ["model_key", "model", "model_name", "checkpoint"])
    rel_col = pick_col(df, ["relation_key", "relation", "task", "property"])
    delta_col = pick_col(df, ["cos_improvement", "deltacos", "delta_cos", "deltaCos", "DeltaCos", "Δcos"])

    if not (model_col and rel_col and delta_col):
        raise ValueError(
            f"[deltacos] Missing required columns in {path}\n"
            f"Columns: {list(df.columns)}\n"
            f"Found model_col={model_col}, rel_col={rel_col}, delta_col={delta_col}"
        )

    # Standardize column names
    if model_col != "model_key":
        df = df.rename(columns={model_col: "model_key"})
        model_col = "model_key"
    if rel_col != "relation_key":
        df = df.rename(columns={rel_col: "relation_key"})
        rel_col = "relation_key"
    if delta_col != "cos_improvement":
        df = df.rename(columns={delta_col: "cos_improvement"})
        delta_col = "cos_improvement"

    # Keep minimal columns + drop duplicates
    df = df[[model_col, rel_col, delta_col]].drop_duplicates()

    return DeltaCosTable(path=path, df=df, model_col=model_col, rel_col=rel_col, delta_col=delta_col)

def autodetect_deltacos(summary_df: pd.DataFrame) -> DeltaCosTable:
    rels = set(summary_df["relation_key"].astype(str))
    models = set(summary_df["model_key"].astype(str))

    # Search likely locations
    roots = ["runs", "data"]
    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            b = os.path.basename(p).lower()
            if ("deltacos" in b) or ("delta" in b and "cos" in b) or ("cos_improvement" in b):
                candidates.append(p)

    scored = []
    for p in candidates:
        try:
            tab = load_deltacos(p)
        except Exception:
            continue

        df = tab.df
        # Score by how many (model,relation) pairs it covers
        m = df["model_key"].astype(str).isin(models) & df["relation_key"].astype(str).isin(rels)
        covered_pairs = int(m.sum())
        covered_models = int(df.loc[m, "model_key"].nunique())
        covered_rels = int(df.loc[m, "relation_key"].nunique())
        if covered_pairs == 0:
            continue

        score = covered_pairs * 10_000 + covered_models * 100 + covered_rels
        mtime = os.path.getmtime(p)
        scored.append((score, mtime, tab))

    if not scored:
        raise RuntimeError(
            "Could not auto-detect a deltacos CSV.\n"
            "Please pass --deltacos /path/to/deltacos.csv\n"
            "Hint: try\n"
            "  find runs data -type f -iname '*deltacos*.csv' -o -iname '*cos_improvement*.csv'\n"
        )

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]

# ---------------- plotting ----------------

def plot_panel(df: pd.DataFrame, y_col: str, out_base: str, y_label: str, dpi: int = 300) -> None:
    global_xmin = float(df["cos_improvement"].min()) - 0.04
    global_xmax = float(df["cos_improvement"].max()) + 0.04
    y_min, y_max = -0.05, 1.05

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, mk in zip(axes, MODEL_ORDER):
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub["cos_improvement"].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        r, p = pearson_r_p_two(x, y)
        slope, intercept = fit_line(x, y)

        ax.scatter(x, y, s=60, edgecolor="black", linewidth=0.6)

        if np.isfinite(slope) and np.isfinite(intercept):
            xx = np.linspace(global_xmin, global_xmax, 200)
            yy = slope * xx + intercept
            ax.plot(xx, yy, linestyle="--", linewidth=1.6, color="black", alpha=0.55)

        title = f"{MODEL_TITLE.get(mk, mk)} (r={r:.3f}"
        if np.isfinite(p):
            title += f", p={p:.4g})"
        else:
            title += ")"
        ax.set_title(title, fontsize=13)

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=11)

    fig.text(0.5, 0.05, "LRE cosine improvement (Δcos)", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.52, y_label, rotation=90, ha="center", va="center", fontsize=14)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.28)
    save_fig(fig, out_base, dpi=dpi)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to fig2_lre21_nat_3way_summary.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--deltacos", default="", help="Optional path to deltacos CSV (auto-detected if empty)")
    ap.add_argument("--n-perm", type=int, default=10000, help="Permutations per model/metric (approx)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    s = pd.read_csv(args.summary)
    need = {
        "model_key","relation_key",
        "n_refusal","n_correct","n_hallucination","n_total",
        "hall_rate_answered","hall_rate_unknown",
    }
    missing = need - set(s.columns)
    if missing:
        raise ValueError(f"[summary] Missing columns: {sorted(missing)}")

    # Load deltacos
    if args.deltacos:
        tab = load_deltacos(args.deltacos)
    else:
        tab = autodetect_deltacos(s)

    print(f"[info] Using deltacos CSV: {tab.path}")

    d = tab.df.copy()
    merged = s.merge(d, on=["model_key","relation_key"], how="left")

    # Validate merge coverage
    miss = merged["cos_improvement"].isna().sum()
    if miss > 0:
        bad = merged[merged["cos_improvement"].isna()][["model_key","relation_key"]].drop_duplicates()
        raise RuntimeError(
            f"[merge] Missing cos_improvement for {miss} rows.\n"
            f"Examples:\n{bad.head(20).to_string(index=False)}\n"
            f"Likely you used the wrong deltacos file. Pass --deltacos explicitly."
        )

    out_merged = os.path.join(args.outdir, "fig2_lre21_nat_3way_plus_deltacos.csv")
    merged.to_csv(out_merged, index=False)
    print(f"[write] {out_merged}")

    # Correlations
    rows = []
    for mk in MODEL_ORDER:
        sub = merged[merged["model_key"] == mk].copy()
        if sub.empty:
            continue

        x = sub["cos_improvement"].to_numpy(dtype=float)

        for metric, y_col, w_col in [
            ("answered", "hall_rate_answered", None),
            ("unknown", "hall_rate_unknown", None),
        ]:
            y = sub[y_col].to_numpy(dtype=float)

            r, p = pearson_r_p_two(x, y)
            rho, p_rho = spearman_r_p_two(x, y)
            p_one, p_two = approx_perm_p(x, y, n_perm=args.n_perm, seed=args.seed)

            # Weight by the denominator of the rate (more stable rate -> larger weight)
            if metric == "answered":
                w = (sub["n_correct"] + sub["n_hallucination"]).to_numpy(dtype=float)
            else:
                w = (sub["n_refusal"] + sub["n_hallucination"]).to_numpy(dtype=float)
            r_w = weighted_pearson_r(x, y, w)

            slope, intercept = fit_line(x, y)

            rows.append({
                "model_key": mk,
                "metric": metric,
                "n_rel": int(sub.shape[0]),
                "pearson_r": r,
                "pearson_p_two": p,
                "spearman_rho": rho,
                "spearman_p_two": p_rho,
                "perm_p_one": p_one,
                "perm_p_two": p_two,
                "r_weighted": r_w,
                "slope": slope,
                "intercept": intercept,
            })

    corr = pd.DataFrame(rows)
    out_corr = os.path.join(args.outdir, "fig2_lre21_nat_3way_corr.csv")
    corr.to_csv(out_corr, index=False)
    print(f"[write] {out_corr}")

    # Plots
    plot_panel(
        merged,
        y_col="hall_rate_answered",
        out_base=os.path.join(args.outdir, "fig2_lre21_nat_3way_scatter_answered_panel_4models"),
        y_label="Hallucination rate (hall / (hall + correct))",
        dpi=args.dpi,
    )
    plot_panel(
        merged,
        y_col="hall_rate_unknown",
        out_base=os.path.join(args.outdir, "fig2_lre21_nat_3way_scatter_unknown_panel_4models"),
        y_label="Hallucination rate (hall / (hall + refusal))",
        dpi=args.dpi,
    )

    print(f"[done] Wrote plots to: {args.outdir}")

if __name__ == "__main__":
    main()
