#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6: Analyze per-triple delta_cos vs behavior labels on the Hernandez LRE dataset.

Inputs:
  (1) Step4 labels.csv.gz (3-way): REFUSAL / CORRECT / HALLUCINATION
  (2) Step5 per-triple delta_cos outputs (gold object) in a directory

Outputs (outdir):
  - merged_per_triple.csv.gz              : merged table (labels + deltacos) AFTER split filtering
  - relation_summary.csv.gz               : per (model, relation) summary stats AFTER split filtering
  - fit_summary.csv.gz                    : per (model, relation) fit metrics (logistic/isotonic/spline) AFTER split filtering
  - plots/
      pooled_{model}.pdf                  : pooled (all relations) binned stairs + fitted curves
      relation_fits_{model}.pdf           : multipage PDF: per relation plot (stairs+CI + fits)
      method_compare_{model}.pdf          : method metric distributions (AUC/logloss) across relations

Default target:
  hallucination_given_value = P(HALLUCINATION | not REFUSAL)

IMPORTANT:
  By default we analyze ONLY split=test (recommended), to avoid in-sample leakage:
    Step5 fits direction on train, so evaluating delta_cos->behavior on train is optimistic.
  Use --split all to reproduce the old behavior.
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

# ---- optional dependency: scikit-learn ----
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, SplineTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
except Exception:
    print("[error] scikit-learn is required for Step6 fits.", file=sys.stderr)
    print("        Install with: pip install -U scikit-learn", file=sys.stderr)
    raise


# ---------------------------
# Helpers: IO / normalization
# ---------------------------

RENAME_MAP = {
    "model_name": "model_key",
    "relation": "relation_key",
    "rel": "relation_key",
    "prompt_id": "id",
    "example_id": "id",
    "idx": "id",
}

DELTA_COS_CANDIDATES = [
    "delta_cos",
    "deltacos",
    "cos_improvement",
    "delta_cos_gold",
    "deltacos_gold",
    "cos_improvement_gold",
]

LABEL_CANDIDATES = ["label_3way", "label", "judge_label_3way", "judge_label"]

def _read_jsonl_any(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_json(path, lines=True, compression="gzip")
    return pd.read_json(path, lines=True)

def _read_csv_any(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

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

def _discover_deltacos_tables(deltacos_dir: str) -> pd.DataFrame:
    """
    Find and load per-triple deltacos data from a directory.
    We concatenate all found tables that have (model_key, id, delta_cos_col).
    """
    if not os.path.isdir(deltacos_dir):
        raise FileNotFoundError(f"deltacos_dir not found: {deltacos_dir}")

    csvs = sorted(glob(os.path.join(deltacos_dir, "**", "*.csv"), recursive=True)) + \
           sorted(glob(os.path.join(deltacos_dir, "**", "*.csv.gz"), recursive=True))
    jsonls = sorted(glob(os.path.join(deltacos_dir, "**", "*.jsonl"), recursive=True)) + \
            sorted(glob(os.path.join(deltacos_dir, "**", "*.jsonl.gz"), recursive=True))

    paths = csvs + jsonls
    if not paths:
        raise FileNotFoundError(f"No csv/jsonl found under: {deltacos_dir}")

    dfs = []
    for p in paths:
        try:
            if p.endswith(".csv") or p.endswith(".csv.gz"):
                df = _read_csv_any(p)
            else:
                df = _read_jsonl_any(p)
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
            f"Could not find any tables with required cols under {deltacos_dir}. "
            f"Need at least (model_key, id, one of {DELTA_COS_CANDIDATES})."
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
    assert x.ndim == 1 and y.ndim == 1 and x.shape[0] == y.shape[0]
    n = x.shape[0]
    if n == 0:
        return []
    n_bins = max(1, min(n_bins, n))

    order = np.argsort(x)
    chunks = np.array_split(order, n_bins)

    bins: List[BinStat] = []
    for idxs in chunks:
        if idxs.size == 0:
            continue
        xs = x[idxs]
        ys = y[idxs]
        x_lo = float(np.min(xs))
        x_hi = float(np.max(xs))
        x_mid = float(np.mean(xs))
        nn = int(ys.size)
        kk = int(np.sum(ys == 1))
        pp = kk / nn if nn else float("nan")
        ci_lo, ci_hi = wilson_ci(kk, nn)
        bins.append(BinStat(x_lo=x_lo, x_hi=x_hi, x_mid=x_mid, n=nn, k=kk, p=float(pp), ci_lo=ci_lo, ci_hi=ci_hi))
    return bins


# ---------------------------
# Fitting (logistic / isotonic / spline)
# ---------------------------

def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)

def _stratified_cv_splits(y: np.ndarray, max_splits: int = 5, seed: int = 0) -> Optional[StratifiedKFold]:
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return None
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    m = min(n0, n1)
    if m < 2:
        return None
    n_splits = min(max_splits, m)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def cv_predict_proba_sklearn(est, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, int]:
    y = np.asarray(y).astype(int)
    n = y.size
    skf = _stratified_cv_splits(y)
    if skf is None:
        return np.full(n, float(np.mean(y))), 0

    p_hat = np.zeros(n, dtype=float)
    for tr, te in skf.split(X, y):
        est_fold = est
        try:
            from sklearn.base import clone
            est_fold = clone(est)
        except Exception:
            pass
        est_fold.fit(X[tr], y[tr])
        p_hat[te] = est_fold.predict_proba(X[te])[:, 1]
    return p_hat, skf.get_n_splits()

def cv_predict_proba_isotonic(x: np.ndarray, y: np.ndarray, increasing: bool) -> Tuple[np.ndarray, int]:
    y = np.asarray(y).astype(int)
    n = y.size
    skf = _stratified_cv_splits(y)
    if skf is None:
        return np.full(n, float(np.mean(y))), 0

    p_hat = np.zeros(n, dtype=float)
    for tr, te in skf.split(x.reshape(-1, 1), y):
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        iso.fit(x[tr], y[tr])
        p_hat[te] = iso.predict(x[te])
    return p_hat, skf.get_n_splits()

def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = _clip_prob(p_pred)
    out = {}
    out["cv_logloss"] = float(log_loss(y_true, p_pred, labels=[0, 1]))
    out["cv_brier"] = float(brier_score_loss(y_true, p_pred))
    if np.unique(y_true).size < 2:
        out["cv_auc"] = float("nan")
    else:
        out["cv_auc"] = float(roc_auc_score(y_true, p_pred))
    return out

def fit_logistic(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    X = x.reshape(-1, 1)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
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

    p_cv, nfold = cv_predict_proba_sklearn(pipe, X, y)
    p_cv = _clip_prob(p_cv)
    metrics = compute_metrics(y, p_cv)

    return {
        "est": pipe,
        "logit_slope": slope_x,
        "logit_intercept": intercept_x,
        "cv_folds": nfold,
        **metrics,
    }

def fit_spline_logistic(x: np.ndarray, y: np.ndarray, n_knots: int = 5, degree: int = 3) -> Dict[str, object]:
    X = x.reshape(-1, 1)
    uniq = int(np.unique(x).size)
    k = min(n_knots, max(3, uniq))
    pipe = Pipeline([
        ("spline", SplineTransformer(n_knots=k, degree=degree, include_bias=False)),
        ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)),
    ])
    pipe.fit(X, y)

    p_cv, nfold = cv_predict_proba_sklearn(pipe, X, y)
    p_cv = _clip_prob(p_cv)
    metrics = compute_metrics(y, p_cv)

    return {
        "est": pipe,
        "spline_knots": k,
        "spline_degree": degree,
        "cv_folds": nfold,
        **metrics,
    }

def fit_isotonic(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    if np.unique(y).size < 2:
        inc = True
    else:
        corr = np.corrcoef(x, y)[0, 1]
        inc = bool(corr >= 0)

    iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
    iso.fit(x, y)

    p_cv, nfold = cv_predict_proba_isotonic(x, y, increasing=inc)
    p_cv = _clip_prob(p_cv)
    metrics = compute_metrics(y, p_cv)

    return {
        "est": iso,
        "isotonic_increasing": inc,
        "cv_folds": nfold,
        **metrics,
    }


# ---------------------------
# Plotting
# ---------------------------

def plot_bins_with_ci(ax, bins: List[BinStat], label: str = "binned") -> None:
    if not bins:
        return
    for b in bins:
        ax.fill_between([b.x_lo, b.x_hi], [b.ci_lo, b.ci_lo], [b.ci_hi, b.ci_hi], alpha=0.20)
        ax.plot([b.x_lo, b.x_hi], [b.p, b.p], linewidth=2.2, label=label if b is bins[0] else None)
    ax.scatter([b.x_mid for b in bins], [b.p for b in bins], s=18)

def plot_fit_curve(ax, est, x_min: float, x_max: float, label: str) -> None:
    grid = np.linspace(x_min, x_max, 250)
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(grid.reshape(-1, 1))[:, 1]
    else:
        p = est.predict(grid)
    p = _clip_prob(np.asarray(p))
    ax.plot(grid, p, linewidth=1.8, label=label)

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, required=True,
                    help="Step4 labels.csv.gz (3-way labels)")
    ap.add_argument("--deltacos_dir", type=str, required=True,
                    help="Step5 output directory containing per-triple delta_cos (gold)")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Output directory for Step6 analysis artifacts")
    ap.add_argument("--n_bins", type=int, default=5,
                    help="Equal-count bins per (model, relation) (default: 5)")
    ap.add_argument("--min_value_n", type=int, default=12,
                    help="Min #value-providing samples per (model, relation) to fit curves")
    ap.add_argument("--max_relations_per_model", type=int, default=999999,
                    help="Safety cap: max relations to plot per model (default: very large)")
    ap.add_argument("--split", type=str, default="test", choices=["test", "train", "all"],
                    help="Which Step5 split to analyze. Default: test (recommended). Use all to reproduce old behavior.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[config] split_filter={args.split}")

    # ---- load labels ----
    print(f"[load] labels: {args.labels_csv}")
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
    print(f"[load] deltacos from dir: {args.deltacos_dir}")
    df_dc = _discover_deltacos_tables(args.deltacos_dir)
    df_dc = _maybe_rename_cols(df_dc)
    if "delta_cos" not in df_dc.columns:
        dc_col = _find_first_existing(df_dc, DELTA_COS_CANDIDATES)
        if dc_col is None:
            raise ValueError(f"deltacos tables missing delta_cos. Need one of {DELTA_COS_CANDIDATES}")
        df_dc = df_dc.rename(columns={dc_col: "delta_cos"})

    # merge keys: prefer (model_key, relation_key, id) if possible
    merge_keys = ["model_key", "id"]
    if "relation_key" in df_dc.columns and "relation_key" in df_lab.columns:
        merge_keys = ["model_key", "relation_key", "id"]

    df_lab = df_lab.drop_duplicates(subset=merge_keys)
    df_dc = df_dc.drop_duplicates(subset=merge_keys)

    print(f"[merge] keys = {merge_keys}")
    df = df_lab.merge(df_dc, on=merge_keys, how="inner", suffixes=("", "_dc"))

    # normalize possible split column name
    if "split" not in df.columns:
        for cand in ["split_dc", "split_x", "split_y"]:
            if cand in df.columns:
                df.rename(columns={cand: "split"}, inplace=True)
                break

    print(f"[merge] merged rows = {len(df)}")

    # numeric delta_cos
    df["delta_cos"] = pd.to_numeric(df["delta_cos"], errors="coerce")
    n_missing_dc = int(df["delta_cos"].isna().sum())
    print(f"[merge] missing/NaN delta_cos = {n_missing_dc}")
    df = df[np.isfinite(df["delta_cos"].to_numpy(dtype=float))].copy()

    # split filter
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError(
                f"Requested --split {args.split} but merged table has no 'split' column. "
                f"Make sure your Step5 per-triple CSV includes split=train/test."
            )
        df["split"] = df["split"].astype(str).str.lower().str.strip()
        df = df[df["split"] == args.split].copy()
        print(f"[filter] after split={args.split}: rows={len(df)}")
    else:
        if "split" in df.columns:
            vc = df["split"].astype(str).str.lower().str.strip().value_counts(dropna=False)
            print(f"[filter] split=all, split counts:\n{vc.to_string()}")

    # Ensure relation metadata exists (from labels side)
    for meta in ["relation_group", "relation_name"]:
        if meta not in df.columns:
            df[meta] = ""

    # derive counts
    df["is_refusal"] = (df["label_3way"] == "REFUSAL")
    df["is_correct"] = (df["label_3way"] == "CORRECT")
    df["is_hall"] = (df["label_3way"] == "HALLUCINATION")
    df["is_value"] = df["is_correct"] | df["is_hall"]

    # Save merged table
    merged_path = os.path.join(args.outdir, "merged_per_triple.csv.gz")
    print(f"[write] {merged_path}")
    df.to_csv(merged_path, index=False, compression="gzip")

    # ---------------------------
    # Relation-level summary table
    # ---------------------------
    grp_cols = ["model_key", "relation_key", "relation_group", "relation_name"]
    rel_rows = []
    for (mk, rk, rg, rn), g in df.groupby(grp_cols):
        n = int(len(g))
        n_ref = int(g["is_refusal"].sum())
        n_val = int(g["is_value"].sum())
        n_hall = int(g["is_hall"].sum())
        n_cor = int(g["is_correct"].sum())

        hall_rate_all = n_hall / n if n else float("nan")
        ref_rate = n_ref / n if n else float("nan")
        val_rate = n_val / n if n else float("nan")

        hall_rate_gv = n_hall / n_val if n_val else float("nan")
        acc_gv = n_cor / n_val if n_val else float("nan")

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
            "delta_cos_mean": float(g["delta_cos"].mean()),
            "delta_cos_std": float(g["delta_cos"].std(ddof=0)),
            "hall_rate_all": hall_rate_all,
            "refusal_rate": ref_rate,
            "value_rate": val_rate,
            "hall_rate_given_value": hall_rate_gv,
            "acc_given_value": acc_gv,
        })
    df_rel = pd.DataFrame(rel_rows)
    rel_path = os.path.join(args.outdir, "relation_summary.csv.gz")
    print(f"[write] {rel_path}")
    df_rel.to_csv(rel_path, index=False, compression="gzip")

    # ---------------------------
    # Per (model, relation) fits
    # Target: P(HALLUCINATION | VALUE)
    # ---------------------------
    fit_rows = []
    print("[fit] per (model, relation) ...")

    for mk in sorted(df["model_key"].unique()):
        df_m = df[df["model_key"] == mk].copy()

        # pooled plot for model
        pooled_pdf = os.path.join(plots_dir, f"pooled_{mk}.pdf")
        with PdfPages(pooled_pdf) as pdf:
            df_val = df_m[df_m["is_value"]].copy()
            x = df_val["delta_cos"].to_numpy(dtype=float)
            y = df_val["is_hall"].to_numpy(dtype=int)

            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            bins = equal_count_bins(x, y, n_bins=max(8, args.n_bins * 2))
            plot_bins_with_ci(ax, bins, label="binned (equal-count)")

            if np.unique(y).size >= 2 and len(y) >= args.min_value_n:
                fit_L = fit_logistic(x, y)
                fit_I = fit_isotonic(x, y)
                fit_S = fit_spline_logistic(x, y)
                x_min, x_max = float(np.min(x)), float(np.max(x))
                plot_fit_curve(ax, fit_L["est"], x_min, x_max, "logistic (CV)")
                plot_fit_curve(ax, fit_I["est"], x_min, x_max, "isotonic (CV)")
                plot_fit_curve(ax, fit_S["est"], x_min, x_max, "spline-logistic (CV)")

                ax.set_title(f"{mk} pooled (split={args.split}) : P(HALLUCINATION | VALUE) vs Δcos\n"
                             f"n_value={len(y)}  hall_rate={float(np.mean(y)):.3f}")
            else:
                ax.set_title(f"{mk} pooled (split={args.split}): insufficient value samples or only one class")
            ax.set_xlabel("Δcos (per-triple, gold object)")
            ax.set_ylabel("P(HALLUCINATION | VALUE)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"[plot] pooled written: {pooled_pdf}")

        # per-relation multipage PDF
        rel_pdf = os.path.join(plots_dir, f"relation_fits_{mk}.pdf")
        with PdfPages(rel_pdf) as pdf:
            rels = sorted(df_m["relation_key"].unique())
            rels = rels[: args.max_relations_per_model]

            for rk in tqdm(rels, desc=f"plots[{mk}] relations"):
                g = df_m[df_m["relation_key"] == rk].copy()
                rg = str(g["relation_group"].iloc[0]) if "relation_group" in g.columns and len(g) else ""
                rn = str(g["relation_name"].iloc[0]) if "relation_name" in g.columns and len(g) else rk

                gv = g[g["is_value"]].copy()
                x = gv["delta_cos"].to_numpy(dtype=float)
                y = gv["is_hall"].to_numpy(dtype=int)

                n_total = int(len(g))
                n_ref = int(g["is_refusal"].sum())
                n_val = int(len(gv))
                n_hall = int(g["is_hall"].sum())
                n_cor = int(g["is_correct"].sum())

                fig, ax = plt.subplots(figsize=(7.6, 4.8))

                if n_val >= 1:
                    bins = equal_count_bins(x, y, n_bins=args.n_bins)
                    plot_bins_with_ci(ax, bins, label="binned (equal-count)")
                    x_min, x_max = float(np.min(x)), float(np.max(x))

                    if n_val >= args.min_value_n and np.unique(y).size >= 2:
                        fit_L = fit_logistic(x, y)
                        fit_I = fit_isotonic(x, y)
                        fit_S = fit_spline_logistic(x, y)

                        plot_fit_curve(ax, fit_L["est"], x_min, x_max, "logistic")
                        plot_fit_curve(ax, fit_I["est"], x_min, x_max, "isotonic")
                        plot_fit_curve(ax, fit_S["est"], x_min, x_max, "spline-logistic")

                        fit_rows.append({
                            "split_filter": args.split,
                            "model_key": mk,
                            "relation_key": rk,
                            "relation_group": rg,
                            "relation_name": rn,
                            "n_total": n_total,
                            "n_refusal": n_ref,
                            "n_value": n_val,
                            "hall_rate_all": n_hall / n_total if n_total else float("nan"),
                            "hall_rate_given_value": n_hall / n_val if n_val else float("nan"),
                            "delta_cos_mean_value": float(np.mean(x)) if len(x) else float("nan"),

                            "logistic_slope": _safe_float(fit_L.get("logit_slope")),
                            "logistic_cv_logloss": _safe_float(fit_L.get("cv_logloss")),
                            "logistic_cv_auc": _safe_float(fit_L.get("cv_auc")),
                            "logistic_cv_brier": _safe_float(fit_L.get("cv_brier")),
                            "logistic_cv_folds": int(fit_L.get("cv_folds", 0)),

                            "isotonic_increasing": bool(fit_I.get("isotonic_increasing", True)),
                            "isotonic_cv_logloss": _safe_float(fit_I.get("cv_logloss")),
                            "isotonic_cv_auc": _safe_float(fit_I.get("cv_auc")),
                            "isotonic_cv_brier": _safe_float(fit_I.get("cv_brier")),
                            "isotonic_cv_folds": int(fit_I.get("cv_folds", 0)),

                            "spline_knots": int(fit_S.get("spline_knots", 0)),
                            "spline_degree": int(fit_S.get("spline_degree", 0)),
                            "spline_cv_logloss": _safe_float(fit_S.get("cv_logloss")),
                            "spline_cv_auc": _safe_float(fit_S.get("cv_auc")),
                            "spline_cv_brier": _safe_float(fit_S.get("cv_brier")),
                            "spline_cv_folds": int(fit_S.get("cv_folds", 0)),
                        })
                    else:
                        fit_rows.append({
                            "split_filter": args.split,
                            "model_key": mk,
                            "relation_key": rk,
                            "relation_group": rg,
                            "relation_name": rn,
                            "n_total": n_total,
                            "n_refusal": n_ref,
                            "n_value": n_val,
                            "hall_rate_all": n_hall / n_total if n_total else float("nan"),
                            "hall_rate_given_value": n_hall / n_val if n_val else float("nan"),
                            "delta_cos_mean_value": float(np.mean(x)) if len(x) else float("nan"),
                            "logistic_slope": float("nan"),
                            "logistic_cv_logloss": float("nan"),
                            "logistic_cv_auc": float("nan"),
                            "logistic_cv_brier": float("nan"),
                            "logistic_cv_folds": 0,
                            "isotonic_increasing": None,
                            "isotonic_cv_logloss": float("nan"),
                            "isotonic_cv_auc": float("nan"),
                            "isotonic_cv_brier": float("nan"),
                            "isotonic_cv_folds": 0,
                            "spline_knots": 0,
                            "spline_degree": 0,
                            "spline_cv_logloss": float("nan"),
                            "spline_cv_auc": float("nan"),
                            "spline_cv_brier": float("nan"),
                            "spline_cv_folds": 0,
                        })

                    ax.set_xlim(x_min - 0.02, x_max + 0.02)
                else:
                    ax.text(0.5, 0.5, "No value-providing samples", ha="center", va="center")
                    ax.set_xlim(0, 1)

                ax.set_title(
                    f"{mk} | {rk} ({rn}) [{rg}] (split={args.split})\n"
                    f"n_total={n_total}  n_ref={n_ref}  n_value={n_val}  "
                    f"hall_rate_gv={ (n_hall / n_val) if n_val else float('nan') :.3f}"
                )
                ax.set_xlabel("Δcos (per-triple, gold object)")
                ax.set_ylabel("P(HALLUCINATION | VALUE)")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best", fontsize=9)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[plot] relation fits written: {rel_pdf}")

        # method comparison plot across relations (for this model)
        fit_df_m = pd.DataFrame([r for r in fit_rows if r["model_key"] == mk])
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        methods = ["logistic", "isotonic", "spline"]
        data = []
        labels = []
        for mname in methods:
            col = f"{mname}_cv_logloss"
            if col in fit_df_m.columns:
                vals = fit_df_m[col]
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    data.append(vals.to_numpy())
                    labels.append(mname)
        if len(data):
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_ylabel("CV log loss (lower is better)")
            ax.set_title(f"{mk}: method comparison across relations (split={args.split})")
            ax.grid(True, axis="y", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No valid fit metrics", ha="center", va="center")
        fig.tight_layout()
        method_pdf = os.path.join(plots_dir, f"method_compare_{mk}.pdf")
        fig.savefig(method_pdf)
        plt.close(fig)
        print(f"[plot] method comparison written: {method_pdf}")

    # save fit summary
    df_fit = pd.DataFrame(fit_rows)
    fit_path = os.path.join(args.outdir, "fit_summary.csv.gz")
    print(f"[write] {fit_path}")
    df_fit.to_csv(fit_path, index=False, compression="gzip")

    # quick console summary: relation-level correlations (per model)
    print("\n== quick correlations across relations (per model) ==")
    for mk in sorted(df_rel["model_key"].unique()):
        sub = df_rel[df_rel["model_key"] == mk].copy()
        sub = sub[np.isfinite(sub["delta_cos_mean"]) & np.isfinite(sub["hall_rate_given_value"])].copy()
        if len(sub) < 5:
            print(f"{mk}: insufficient relations for correlation")
            continue
        r = np.corrcoef(sub["delta_cos_mean"].to_numpy(), sub["hall_rate_given_value"].to_numpy())[0, 1]
        print(f"{mk}: corr(mean Δcos, hall_rate_given_value) = {r:.4f}   (n_rel={len(sub)})")

    print(f"\n[done] step6 outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
