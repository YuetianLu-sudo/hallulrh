#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6 (logistic-only): Analyze per-triple Δcos vs behavior labels on Hernandez LRE (natural, gold).

Inputs:
  (1) Step4 labels.csv.gz (3-way): REFUSAL / CORRECT / HALLUCINATION
  (2) Step5 per-triple delta_cos outputs (gold object) in a directory (prefer per_triple.csv(.gz))

Leakage note:
  Step5 fits relation direction on train split.
  Therefore Step6 defaults to analyzing ONLY split=test (recommended).

Targets (choose one via --target):
  - hall_given_value (PRIMARY):
      subset = VALUE = {CORRECT, HALLUCINATION}
      y = 1[HALLUCINATION]
      => P(HALLUCINATION | VALUE)

  - refusal (SUPPLEMENT):
      subset = ALL
      y = 1[REFUSAL]
      => P(REFUSAL)

  - hall_given_noncorrect (BRIDGE to synthetic):
      subset = NOT CORRECT = {REFUSAL, HALLUCINATION}
      y = 1[HALLUCINATION]
      => P(HALLUCINATION | NOT CORRECT)
      (When synthetic has no correct answers, this mirrors the refusal vs hallucination setup.)

Outputs (outdir):
  - merged_per_triple.csv.gz         : merged table (labels + deltacos) AFTER split filtering
  - relation_summary.csv.gz          : per (model, relation) descriptive stats
  - fit_summary.csv.gz               : per (model, relation) logistic fit metrics for chosen target
  - plots/<target>/
      pooled_{model}.pdf             : pooled binned stairs + Wilson CI + logistic curve (if fit-able)
      relation_fits_{model}.pdf      : multipage per-relation plots (stairs+CI + logistic curve if fit-able)

Console prints:
  - per-model correlation across relations:
      corr(mean Δcos|VALUE, hall_rate_given_value)
    (computed from relation_summary after split filtering)
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
except Exception:
    print("[error] scikit-learn is required. Install: pip install -U scikit-learn", file=sys.stderr)
    raise


RENAME_MAP = {
    "model_name": "model_key",
    "relation": "relation_key",
    "rel": "relation_key",
    "prompt_id": "id",
    "example_id": "id",
    "idx": "id",
}

LABEL_CANDIDATES = ["label_3way", "label", "judge_label_3way", "judge_label"]
DELTA_COS_CANDIDATES = ["delta_cos", "deltacos", "cos_improvement", "delta_cos_gold", "deltacos_gold"]


# ---------------------------
# Helpers: IO / normalization
# ---------------------------

def _maybe_rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = set(df.columns)
    for src, dst in RENAME_MAP.items():
        if dst not in cols and src in cols:
            df.rename(columns={src: dst}, inplace=True)
    return df

def _find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _read_table_any(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
        return pd.read_json(path, lines=True, compression="gzip" if path.endswith(".gz") else None)
    return pd.read_csv(path)

def _discover_deltacos_tables(deltacos_dir: str) -> pd.DataFrame:
    """
    Prefer reading per_triple.csv(.gz) to avoid duplicates from per_relation shards.
    """
    if not os.path.isdir(deltacos_dir):
        raise FileNotFoundError(f"deltacos_dir not found: {deltacos_dir}")

    preferred = []
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.csv"), recursive=True))
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.csv.gz"), recursive=True))
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.jsonl"), recursive=True))
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.jsonl.gz"), recursive=True))

    paths = preferred

    # Fallback only if no per_triple exists
    if not paths:
        all_csv = sorted(glob(os.path.join(deltacos_dir, "**", "*.csv"), recursive=True)) + \
                  sorted(glob(os.path.join(deltacos_dir, "**", "*.csv.gz"), recursive=True))
        all_jsonl = sorted(glob(os.path.join(deltacos_dir, "**", "*.jsonl"), recursive=True)) + \
                    sorted(glob(os.path.join(deltacos_dir, "**", "*.jsonl.gz"), recursive=True))
        paths = all_csv + all_jsonl

        filtered = []
        for p in paths:
            bn = os.path.basename(p)
            if "per_relation" in p.replace("\\", "/"):
                continue
            if bn.startswith("relation_summary") or bn.startswith("fit_summary"):
                continue
            filtered.append(p)
        paths = filtered

    if not paths:
        raise FileNotFoundError(f"No per_triple / csv/jsonl found under: {deltacos_dir}")

    dfs = []
    for p in paths:
        try:
            df = _read_table_any(p)
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}", file=sys.stderr)
            continue

        df = _maybe_rename_cols(df)
        if "model_key" not in df.columns or "id" not in df.columns:
            continue

        dc_col = _find_first_existing(df, DELTA_COS_CANDIDATES)
        if dc_col is None:
            continue
        if dc_col != "delta_cos":
            df = df.rename(columns={dc_col: "delta_cos"})

        dfs.append(df)

    if not dfs:
        raise ValueError(
            f"Could not find any deltacos tables with required cols under {deltacos_dir}. "
            f"Need at least (model_key, id, delta_cos)."
        )

    out = pd.concat(dfs, ignore_index=True)
    return out


# ---------------------------
# Stats: binning + Wilson CI
# ---------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt(phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

@dataclass
class BinStat:
    x_lo: float
    x_hi: float
    x_mid: float
    n: int
    k: int
    p: float
    ci_lo: float
    ci_hi: float

def equal_count_bins(x: np.ndarray, y: np.ndarray, n_bins: int) -> List[BinStat]:
    """
    Robust equal-count binning (works when qcut would fail due to duplicates).
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    assert x.shape[0] == y.shape[0]
    n = x.shape[0]
    if n == 0:
        return []
    n_bins = max(1, min(int(n_bins), n))

    order = np.argsort(x)
    chunks = np.array_split(order, n_bins)

    bins: List[BinStat] = []
    for idxs in chunks:
        if idxs.size == 0:
            continue
        xs = x[idxs]
        ys = y[idxs]
        nn = int(ys.size)
        kk = int(np.sum(ys == 1))
        pp = kk / nn if nn else float("nan")
        ci_lo, ci_hi = wilson_ci(kk, nn)
        bins.append(BinStat(
            x_lo=float(np.min(xs)),
            x_hi=float(np.max(xs)),
            x_mid=float(np.mean(xs)),
            n=nn, k=kk, p=float(pp), ci_lo=ci_lo, ci_hi=ci_hi
        ))
    return bins

def plot_bins_with_ci(ax, bins: List[BinStat], label: str = "binned") -> None:
    if not bins:
        return
    for i, b in enumerate(bins):
        ax.fill_between([b.x_lo, b.x_hi], [b.ci_lo, b.ci_lo], [b.ci_hi, b.ci_hi], alpha=0.20)
        ax.plot([b.x_lo, b.x_hi], [b.p, b.p], linewidth=2.2, label=label if i == 0 else None)
    ax.scatter([b.x_mid for b in bins], [b.p for b in bins], s=18)


# ---------------------------
# Logistic fit + CV metrics
# ---------------------------

def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)

def _stratified_cv_splits(y: np.ndarray, max_splits: int = 5, seed: int = 0) -> Optional[StratifiedKFold]:
    y = np.asarray(y).astype(int)
    if np.unique(y).size < 2:
        return None
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    m = min(n0, n1)
    if m < 2:
        return None
    n_splits = min(max_splits, m)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p_pred = _clip_prob(p_pred)
    out = {
        "cv_logloss": float(log_loss(y_true, p_pred, labels=[0, 1])),
        "cv_brier": float(brier_score_loss(y_true, p_pred)),
    }
    if np.unique(y_true).size < 2:
        out["cv_auc"] = float("nan")
    else:
        out["cv_auc"] = float(roc_auc_score(y_true, p_pred))
    return out

def cv_predict_proba_sklearn(est, X: np.ndarray, y: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, int]:
    """
    Manual CV predict_proba; returns (p_hat, nfolds_used).
    Falls back to constant mean probability if CV not possible.
    """
    y = np.asarray(y).astype(int)
    n = y.size
    skf = _stratified_cv_splits(y, seed=seed)
    if skf is None:
        return np.full(n, float(np.mean(y))), 0

    p_hat = np.zeros(n, dtype=float)
    for tr, te in skf.split(X, y):
        try:
            from sklearn.base import clone
            est_fold = clone(est)
        except Exception:
            est_fold = est
        est_fold.fit(X[tr], y[tr])
        p_hat[te] = est_fold.predict_proba(X[te])[:, 1]
    return p_hat, skf.get_n_splits()

def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, object]:
    """
    Fit logistic on full data for curve, compute CV metrics on out-of-fold probabilities.
    Also report slope/intercept on original x scale (not standardized).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    X = x.reshape(-1, 1)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)),
    ])
    pipe.fit(X, y)

    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    b1 = float(clf.coef_.ravel()[0])
    b0 = float(clf.intercept_.ravel()[0])
    scale = float(scaler.scale_.ravel()[0])
    mean = float(scaler.mean_.ravel()[0])

    slope_x = b1 / scale
    intercept_x = b0 - (b1 * mean / scale)

    p_cv, nfold = cv_predict_proba_sklearn(pipe, X, y, seed=seed)
    metrics = compute_metrics(y, p_cv)

    return {
        "est": pipe,
        "logit_slope": float(slope_x),
        "logit_intercept": float(intercept_x),
        "cv_folds": int(nfold),
        **metrics,
    }

def plot_fit_curve(ax, est, x_min: float, x_max: float, label: str) -> None:
    grid = np.linspace(x_min, x_max, 250)
    p = est.predict_proba(grid.reshape(-1, 1))[:, 1]
    p = _clip_prob(p)
    ax.plot(grid, p, linewidth=1.8, label=label)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--deltacos_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--split", type=str, default="test", choices=["test", "train", "all"],
                    help="Which Step5 split to analyze. Default: test (recommended).")
    ap.add_argument("--target", type=str, default="hall_given_value",
                    choices=["hall_given_value", "refusal", "hall_given_noncorrect"],
                    help="Binary target to fit/plot.")
    ap.add_argument("--n_bins", type=int, default=5,
                    help="Equal-count bins per (model, relation).")
    ap.add_argument("--min_fit_n", type=int, default=30,
                    help="Min #samples in the fitting subset (after split+target filtering) to fit logistic.")
    ap.add_argument("--max_relations_per_model", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots", args.target)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[config] split={args.split} target={args.target} n_bins={args.n_bins} min_fit_n={args.min_fit_n}")

    # ---- load labels ----
    df_lab = pd.read_csv(args.labels_csv)
    df_lab = _maybe_rename_cols(df_lab)

    label_col = _find_first_existing(df_lab, LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"labels_csv missing label column. Need one of: {LABEL_CANDIDATES}")
    if label_col != "label_3way":
        df_lab = df_lab.rename(columns={label_col: "label_3way"})

    need_cols = ["model_key", "id", "relation_key", "label_3way"]
    for c in need_cols:
        if c not in df_lab.columns:
            raise ValueError(f"labels_csv missing required column: {c}. Found: {list(df_lab.columns)}")

    df_lab["label_3way"] = df_lab["label_3way"].astype(str).str.upper().str.strip()
    df_lab["label_3way"] = df_lab["label_3way"].replace({"WRONG": "HALLUCINATION"})

    # ---- load deltacos ----
    df_dc = _discover_deltacos_tables(args.deltacos_dir)
    df_dc = _maybe_rename_cols(df_dc)

    dc_col = _find_first_existing(df_dc, DELTA_COS_CANDIDATES)
    if dc_col is None:
        raise ValueError(f"deltacos tables missing delta_cos. Need one of {DELTA_COS_CANDIDATES}")
    if dc_col != "delta_cos":
        df_dc = df_dc.rename(columns={dc_col: "delta_cos"})

    # ---- merge ----
    merge_keys = ["model_key", "id"]
    if "relation_key" in df_dc.columns and "relation_key" in df_lab.columns:
        merge_keys = ["model_key", "relation_key", "id"]

    df_lab = df_lab.drop_duplicates(subset=merge_keys)
    df_dc = df_dc.drop_duplicates(subset=merge_keys)

    df = df_lab.merge(df_dc, on=merge_keys, how="inner", suffixes=("", "_dc"))
    print(f"[merge] keys={merge_keys} rows={len(df)}")

    # normalize split column
    if "split" not in df.columns:
        for cand in ["split_dc", "split_x", "split_y"]:
            if cand in df.columns:
                df.rename(columns={cand: "split"}, inplace=True)
                break

    df["delta_cos"] = pd.to_numeric(df["delta_cos"], errors="coerce")
    df = df[np.isfinite(df["delta_cos"].to_numpy(dtype=float))].copy()

    # split filter
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("Requested split filtering but merged df has no 'split' column.")
        df["split"] = df["split"].astype(str).str.lower().str.strip()
        df = df[df["split"] == args.split].copy()
        print(f"[filter] split={args.split} rows={len(df)}")
    else:
        if "split" in df.columns:
            vc = df["split"].astype(str).str.lower().str.strip().value_counts(dropna=False)
            print(f"[filter] split=all, split counts:\n{vc.to_string()}")

    # ensure relation metadata exists
    for meta in ["relation_group", "relation_name"]:
        if meta not in df.columns:
            df[meta] = ""

    # derive flags
    df["is_refusal"] = (df["label_3way"] == "REFUSAL")
    df["is_correct"] = (df["label_3way"] == "CORRECT")
    df["is_hall"] = (df["label_3way"] == "HALLUCINATION")
    df["is_value"] = df["is_correct"] | df["is_hall"]
    df["is_noncorrect"] = df["is_refusal"] | df["is_hall"]  # bridge subset

    # save merged
    merged_path = os.path.join(args.outdir, "merged_per_triple.csv.gz")
    df.to_csv(merged_path, index=False, compression="gzip")
    print(f"[write] {merged_path}")

    # ---------------------------
    # relation summary (descriptive)
    # ---------------------------
    grp_cols = ["model_key", "relation_key", "relation_group", "relation_name"]
    rel_rows = []
    for (mk, rk, rg, rn), g in df.groupby(grp_cols):
        n = int(len(g))
        n_ref = int(g["is_refusal"].sum())
        n_val = int(g["is_value"].sum())
        n_non = int(g["is_noncorrect"].sum())
        n_hall = int(g["is_hall"].sum())
        n_cor = int(g["is_correct"].sum())

        rel_rows.append({
            "split_filter": args.split,
            "model_key": mk,
            "relation_key": rk,
            "relation_group": rg,
            "relation_name": rn,

            "n_total": n,
            "n_refusal": n_ref,
            "n_value": n_val,
            "n_correct": n_cor,
            "n_hallucination": n_hall,
            "n_noncorrect": n_non,

            "refusal_rate": (n_ref / n) if n else float("nan"),
            "value_rate": (n_val / n) if n else float("nan"),
            "hall_rate_all": (n_hall / n) if n else float("nan"),

            "hall_rate_given_value": (n_hall / n_val) if n_val else float("nan"),
            "acc_given_value": (n_cor / n_val) if n_val else float("nan"),

            # bridge metric
            "hall_rate_given_noncorrect": (n_hall / n_non) if n_non else float("nan"),

            "delta_cos_mean_all": float(g["delta_cos"].mean()) if n else float("nan"),
            "delta_cos_std_all": float(g["delta_cos"].std(ddof=0)) if n else float("nan"),

            "delta_cos_mean_value": float(g.loc[g["is_value"], "delta_cos"].mean()) if n_val else float("nan"),
            "delta_cos_mean_correct": float(g.loc[g["is_correct"], "delta_cos"].mean()) if n_cor else float("nan"),
            "delta_cos_mean_hall": float(g.loc[g["is_hall"], "delta_cos"].mean()) if n_hall else float("nan"),
            "delta_cos_mean_refusal": float(g.loc[g["is_refusal"], "delta_cos"].mean()) if n_ref else float("nan"),
            "delta_cos_mean_noncorrect": float(g.loc[g["is_noncorrect"], "delta_cos"].mean()) if n_non else float("nan"),
        })

    df_rel = pd.DataFrame(rel_rows)
    rel_path = os.path.join(args.outdir, "relation_summary.csv.gz")
    df_rel.to_csv(rel_path, index=False, compression="gzip")
    print(f"[write] {rel_path}")

    # ---------------------------
    # Print the correlations you care about (per model, across relations)
    # ---------------------------
    print("\n== per-model correlation across relations ==")
    for mk in sorted(df_rel["model_key"].unique()):
        sub = df_rel[df_rel["model_key"] == mk].copy()
        sub = sub[np.isfinite(sub["delta_cos_mean_value"]) & np.isfinite(sub["hall_rate_given_value"])].copy()
        # keep only relations with at least 1 VALUE example
        sub = sub[sub["n_value"] > 0].copy()
        if len(sub) < 5:
            print(f"{mk}: insufficient relations for correlation (n_rel={len(sub)})")
            continue

        r = np.corrcoef(sub["delta_cos_mean_value"].to_numpy(dtype=float),
                        sub["hall_rate_given_value"].to_numpy(dtype=float))[0, 1]
        print(f"{mk}: Pearson r(mean Δcos|VALUE, hall_rate_given_value) = {r:.4f} (n_rel={len(sub)})")

    # ---------------------------
    # Choose subset + target
    # ---------------------------
    def subset_and_target(g: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
        if args.target == "hall_given_value":
            gg = g[g["is_value"]].copy()
            x = gg["delta_cos"].to_numpy(dtype=float)
            y = gg["is_hall"].to_numpy(dtype=int)
            ylabel = "P(HALLUCINATION | VALUE)"
        elif args.target == "hall_given_noncorrect":
            gg = g[g["is_noncorrect"]].copy()
            x = gg["delta_cos"].to_numpy(dtype=float)
            y = gg["is_hall"].to_numpy(dtype=int)  # among {hall,ref}, 1=hall
            ylabel = "P(HALLUCINATION | NOT CORRECT)"
        else:  # refusal
            x = g["delta_cos"].to_numpy(dtype=float)
            y = g["is_refusal"].to_numpy(dtype=int)
            ylabel = "P(REFUSAL)"
        return x, y, ylabel

    # ---------------------------
    # Fit + plot (logistic only)
    # ---------------------------
    fit_rows = []
    print("\n[fit+plot] per model ...")

    for mk in sorted(df["model_key"].unique()):
        df_m = df[df["model_key"] == mk].copy()

        # pooled plot
        pooled_pdf = os.path.join(plots_dir, f"pooled_{mk}.pdf")
        with PdfPages(pooled_pdf) as pdf:
            x, y, ylabel = subset_and_target(df_m)

            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            if len(x) >= 1:
                bins = equal_count_bins(x, y, n_bins=max(8, args.n_bins * 2))
                plot_bins_with_ci(ax, bins, label="binned (equal-count)")
                x_min, x_max = float(np.min(x)), float(np.max(x))

                if len(x) >= args.min_fit_n and np.unique(y).size >= 2:
                    fit = fit_logistic(x, y, seed=args.seed)
                    plot_fit_curve(ax, fit["est"], x_min, x_max, "logistic (fit; CV metrics)")
                    title = (f"{mk} pooled split={args.split} target={args.target}\n"
                             f"n={len(y)} base_rate={float(np.mean(y)):.3f} "
                             f"slope={fit['logit_slope']:.3f} cv_logloss={fit['cv_logloss']:.3f}")
                else:
                    title = (f"{mk} pooled split={args.split} target={args.target}\n"
                             f"n={len(y)} base_rate={float(np.mean(y)):.3f} (no fit: small n or single class)")
                ax.set_xlim(x_min - 0.02, x_max + 0.02)
            else:
                title = f"{mk} pooled: no samples for target subset"

            ax.set_title(title)
            ax.set_xlabel("Δcos (per-triple, gold object)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"[plot] pooled: {pooled_pdf}")

        # per-relation multipage PDF
        rel_pdf = os.path.join(plots_dir, f"relation_fits_{mk}.pdf")
        with PdfPages(rel_pdf) as pdf:
            rels = sorted(df_m["relation_key"].unique())[: args.max_relations_per_model]

            for rk in tqdm(rels, desc=f"relations[{mk}]"):
                g = df_m[df_m["relation_key"] == rk].copy()
                rg = str(g["relation_group"].iloc[0]) if len(g) else ""
                rn = str(g["relation_name"].iloc[0]) if len(g) else rk

                x, y, ylabel = subset_and_target(g)

                n_total = int(len(g))
                n_ref = int(g["is_refusal"].sum())
                n_val = int(g["is_value"].sum())
                n_non = int(g["is_noncorrect"].sum())
                n_hall = int(g["is_hall"].sum())
                n_cor = int(g["is_correct"].sum())

                fig, ax = plt.subplots(figsize=(7.6, 4.8))

                row = {
                    "split_filter": args.split,
                    "target": args.target,
                    "model_key": mk,
                    "relation_key": rk,
                    "relation_group": rg,
                    "relation_name": rn,
                    "n_total": n_total,
                    "n_refusal": n_ref,
                    "n_value": n_val,
                    "n_noncorrect": n_non,
                    "n_correct": n_cor,
                    "n_hallucination": n_hall,
                    "subset_n": int(len(y)),
                    "subset_base_rate": float(np.mean(y)) if len(y) else float("nan"),
                }

                if len(x) >= 1:
                    bins = equal_count_bins(x, y, n_bins=args.n_bins)
                    plot_bins_with_ci(ax, bins, label="binned (equal-count)")
                    x_min, x_max = float(np.min(x)), float(np.max(x))
                    ax.set_xlim(x_min - 0.02, x_max + 0.02)

                    if len(x) >= args.min_fit_n and np.unique(y).size >= 2:
                        fit = fit_logistic(x, y, seed=args.seed)
                        plot_fit_curve(ax, fit["est"], x_min, x_max, "logistic")

                        row.update({
                            "logistic_slope": float(fit["logit_slope"]),
                            "logistic_intercept": float(fit["logit_intercept"]),
                            "cv_folds": int(fit["cv_folds"]),
                            "cv_logloss": float(fit["cv_logloss"]),
                            "cv_auc": float(fit["cv_auc"]),
                            "cv_brier": float(fit["cv_brier"]),
                        })
                    else:
                        row.update({
                            "logistic_slope": float("nan"),
                            "logistic_intercept": float("nan"),
                            "cv_folds": 0,
                            "cv_logloss": float("nan"),
                            "cv_auc": float("nan"),
                            "cv_brier": float("nan"),
                        })
                else:
                    ax.text(0.5, 0.5, "No samples for this target subset", ha="center", va="center")
                    row.update({
                        "logistic_slope": float("nan"),
                        "logistic_intercept": float("nan"),
                        "cv_folds": 0,
                        "cv_logloss": float("nan"),
                        "cv_auc": float("nan"),
                        "cv_brier": float("nan"),
                    })

                fit_rows.append(row)

                # Title includes the key counts for interpretation
                ax.set_title(
                    f"{mk} | {rk} ({rn}) [{rg}] split={args.split} target={args.target}\n"
                    f"n_total={n_total} n_ref={n_ref} n_value={n_val} n_noncorrect={n_non}"
                )
                ax.set_xlabel("Δcos (per-triple, gold object)")
                ax.set_ylabel(ylabel)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best", fontsize=9)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[plot] relation fits: {rel_pdf}")

    df_fit = pd.DataFrame(fit_rows)
    fit_path = os.path.join(args.outdir, "fit_summary.csv.gz")
    df_fit.to_csv(fit_path, index=False, compression="gzip")
    print(f"[write] {fit_path}")

    print(f"\n[done] outdir={args.outdir}")
    print(f"[done] plots_dir={plots_dir}")

if __name__ == "__main__":
    main()
