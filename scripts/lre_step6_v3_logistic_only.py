#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step6-v3 (logistic-only): Analyze per-triple Δcos vs behavior labels on Hernandez LRE (natural, gold).

Key design choices (paper-friendly):
  1) Default split=test to avoid leakage:
       Step5 learns relation direction on train; evaluating Δcos→behavior on train is optimistic.
  2) Support multiple binary targets (run once, get all):
       - hall_given_value      : P(HALLUCINATION | VALUE) where VALUE = CORRECT or HALLUCINATION
       - hall_given_noncorrect : P(HALLUCINATION | NONCORRECT) where NONCORRECT = REFUSAL or HALLUCINATION
       - refusal               : P(REFUSAL)
  3) Fit ONLY logistic regression (interpretable + stable), with stratified CV metrics.
  4) Report per-model correlations across relations for each target:
       Pearson r(mean Δcos|subset, rate)
     We print three versions to diagnose "why r changed":
       - r_all      : all relations (equal weight per relation)
       - r_stable   : only relations with denominator >= --min_rel_denom (default 12)
       - r_weighted : weighted Pearson across relations (weights = denominator)
     This mitigates noise from tiny test-set denominators (e.g., n_value=3 makes rates extremely noisy).

Outputs (outdir):
  - merged_per_triple.csv.gz        : merged labels + Δcos after split filtering
  - relation_summary.csv.gz         : per (model, relation) descriptive stats (all targets' denominators/rates/means)
  - fit_summary.csv.gz              : per (target, model, relation) logistic fit metrics (includes include_rel flag)
  - correlation_summary.csv         : per (target, model) correlations across relations (all/stable/weighted)
  - plots/<target>/
      pooled_<model>.pdf            : pooled stairs+CI + logistic curve (if enough samples)
      relation_fits_<model>.pdf     : multipage per-relation stairs+CI (+ logistic when enough samples),
                                      only for relations with denom>=min_rel_denom (to reduce noise/pages)
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


# ---------------------------
# Column normalization helpers
# ---------------------------

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
    Prefer reading per_triple.* to avoid duplication from per_relation shards.
    """
    if not os.path.isdir(deltacos_dir):
        raise FileNotFoundError(f"deltacos_dir not found: {deltacos_dir}")

    preferred = []
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.csv"), recursive=True))
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.csv.gz"), recursive=True))
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.jsonl"), recursive=True))
    preferred += sorted(glob(os.path.join(deltacos_dir, "**", "per_triple.jsonl.gz"), recursive=True))

    paths = preferred

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
# Stats: equal-count bins + Wilson CI
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
    n_bins = max(1, min(int(n_bins), n))
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


def plot_bins_with_ci(ax, bins: List[BinStat], label: str = "binned") -> None:
    if not bins:
        return
    for b in bins:
        ax.fill_between([b.x_lo, b.x_hi], [b.ci_lo, b.ci_lo], [b.ci_hi, b.ci_hi], alpha=0.20)
        ax.plot([b.x_lo, b.x_hi], [b.p, b.p], linewidth=2.2, label=label if b is bins[0] else None)
    ax.scatter([b.x_mid for b in bins], [b.p for b in bins], s=18)


# ---------------------------
# Logistic + stratified CV metrics
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


def cv_predict_proba_logistic(x: np.ndarray, y: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, int]:
    """
    CV predict_proba for a logistic pipeline (StandardScaler + LogisticRegression).
    Returns (p_hat, n_folds_used). If CV not possible, returns constant probabilities.
    """
    y = np.asarray(y).astype(int)
    x = np.asarray(x).astype(float)
    n = y.size
    skf = _stratified_cv_splits(y, seed=seed)
    if skf is None:
        return np.full(n, float(np.mean(y))), 0

    p_hat = np.zeros(n, dtype=float)
    for tr, te in skf.split(x.reshape(-1, 1), y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)),
        ])
        pipe.fit(x[tr].reshape(-1, 1), y[tr])
        p_hat[te] = pipe.predict_proba(x[te].reshape(-1, 1))[:, 1]
    return p_hat, skf.get_n_splits()


def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p_pred = _clip_prob(np.asarray(p_pred, dtype=float))
    out = {
        "cv_logloss": float(log_loss(y_true, p_pred, labels=[0, 1])),
        "cv_brier": float(brier_score_loss(y_true, p_pred)),
    }
    if np.unique(y_true).size < 2:
        out["cv_auc"] = float("nan")
    else:
        out["cv_auc"] = float(roc_auc_score(y_true, p_pred))
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, object]:
    """
    Fit logistic on full data; report slope on original x scale and CV metrics.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
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

    p_cv, nfold = cv_predict_proba_logistic(x, y, seed=seed)
    metrics = compute_metrics(y, p_cv)

    return {
        "est": pipe,
        "logit_slope": slope_x,
        "logit_intercept": intercept_x,
        "cv_folds": int(nfold),
        **metrics,
    }


def plot_logistic_curve(ax, est, x_min: float, x_max: float, label: str) -> None:
    grid = np.linspace(x_min, x_max, 250)
    p = est.predict_proba(grid.reshape(-1, 1))[:, 1]
    p = _clip_prob(np.asarray(p))
    ax.plot(grid, p, linewidth=1.8, label=label)


# ---------------------------
# Correlation utilities
# ---------------------------

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def weighted_pearson_r(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[m]; y = y[m]; w = w[m]
    if x.size < 3:
        return float("nan")
    wsum = float(w.sum())
    mx = float((w * x).sum() / wsum)
    my = float((w * y).sum() / wsum)
    cov = float((w * (x - mx) * (y - my)).sum() / wsum)
    vx = float((w * (x - mx) ** 2).sum() / wsum)
    vy = float((w * (y - my) ** 2).sum() / wsum)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / math.sqrt(vx * vy))


# ---------------------------
# Target definitions
# ---------------------------

TARGETS = ["hall_given_value", "hall_given_noncorrect", "refusal"]

def subset_and_target(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, str, str, str]:
    """
    Returns (x, y, ylabel, denom_col, mean_col)
    """
    if target == "hall_given_value":
        sub = df[df["is_value"]].copy()
        x = sub["delta_cos"].to_numpy(dtype=float)
        y = sub["is_hall"].to_numpy(dtype=int)
        ylabel = "P(HALLUCINATION | VALUE)"
        denom_col = "n_value"
        mean_col = "delta_cos_mean_value"
    elif target == "hall_given_noncorrect":
        sub = df[df["is_noncorrect"]].copy()
        x = sub["delta_cos"].to_numpy(dtype=float)
        y = sub["is_hall"].to_numpy(dtype=int)
        ylabel = "P(HALLUCINATION | NONCORRECT)"
        denom_col = "n_noncorrect"
        mean_col = "delta_cos_mean_noncorrect"
    elif target == "refusal":
        x = df["delta_cos"].to_numpy(dtype=float)
        y = df["is_refusal"].to_numpy(dtype=int)
        ylabel = "P(REFUSAL)"
        denom_col = "n_total"
        mean_col = "delta_cos_mean_all"
    else:
        raise ValueError(f"Unknown target: {target}")
    return x, y, ylabel, denom_col, mean_col


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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
    ap.add_argument("--targets", type=str, default="hall_given_value",
                    help="Comma-separated targets: hall_given_value,hall_given_noncorrect,refusal or 'all'.")
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--min_fit_n", type=int, default=30,
                    help="Min #samples in the target subset (per relation) to fit logistic + report CV metrics.")
    ap.add_argument("--min_rel_denom", type=int, default=12,
                    help="Min denominator size to (a) include relation in correlation summary, and (b) include in relation_fits PDF. Set 0 to include all.")
    ap.add_argument("--max_relations_per_model", type=int, default=999999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # parse targets
    tgt_arg = args.targets.strip().lower()
    if tgt_arg == "all":
        targets = TARGETS
    else:
        targets = [t.strip().lower() for t in tgt_arg.split(",") if t.strip()]
    for t in targets:
        if t not in TARGETS:
            raise ValueError(f"Unknown target '{t}'. Allowed: {TARGETS} or 'all'.")

    print(f"[config] split={args.split} targets={targets} n_bins={args.n_bins} "
          f"min_fit_n={args.min_fit_n} min_rel_denom={args.min_rel_denom} seed={args.seed}")

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

    if "delta_cos" not in df_dc.columns:
        dc_col = _find_first_existing(df_dc, DELTA_COS_CANDIDATES)
        if dc_col is None:
            raise ValueError(f"deltacos tables missing delta_cos. Need one of {DELTA_COS_CANDIDATES}")
        df_dc = df_dc.rename(columns={dc_col: "delta_cos"})

    # merge keys
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
    df["is_noncorrect"] = df["is_refusal"] | df["is_hall"]

    # write merged
    merged_path = os.path.join(args.outdir, "merged_per_triple.csv.gz")
    df.to_csv(merged_path, index=False, compression="gzip")
    print(f"[write] {merged_path}")

    # ---------------------------
    # Relation-level summary (descriptive, covers ALL targets)
    # ---------------------------
    grp_cols = ["model_key", "relation_key", "relation_group", "relation_name"]
    rel_rows = []
    for (mk, rk, rg, rn), g in df.groupby(grp_cols):
        n_total = int(len(g))
        n_ref = int(g["is_refusal"].sum())
        n_cor = int(g["is_correct"].sum())
        n_hall = int(g["is_hall"].sum())
        n_value = int(g["is_value"].sum())
        n_noncorrect = int(g["is_noncorrect"].sum())

        rel_rows.append({
            "split_filter": args.split,
            "model_key": mk,
            "relation_key": rk,
            "relation_group": rg,
            "relation_name": rn,

            "n_total": n_total,
            "n_refusal": n_ref,
            "n_correct": n_cor,
            "n_hallucination": n_hall,
            "n_value": n_value,
            "n_noncorrect": n_noncorrect,

            "refusal_rate": (n_ref / n_total) if n_total else float("nan"),
            "hall_rate_all": (n_hall / n_total) if n_total else float("nan"),
            "hall_rate_given_value": (n_hall / n_value) if n_value else float("nan"),
            "acc_given_value": (n_cor / n_value) if n_value else float("nan"),
            "hall_rate_given_noncorrect": (n_hall / n_noncorrect) if n_noncorrect else float("nan"),

            "delta_cos_mean_all": float(g["delta_cos"].mean()),
            "delta_cos_std_all": float(g["delta_cos"].std(ddof=0)),

            "delta_cos_mean_value": float(g.loc[g["is_value"], "delta_cos"].mean()) if n_value else float("nan"),
            "delta_cos_mean_noncorrect": float(g.loc[g["is_noncorrect"], "delta_cos"].mean()) if n_noncorrect else float("nan"),
            "delta_cos_mean_correct": float(g.loc[g["is_correct"], "delta_cos"].mean()) if n_cor else float("nan"),
            "delta_cos_mean_hall": float(g.loc[g["is_hall"], "delta_cos"].mean()) if n_hall else float("nan"),
            "delta_cos_mean_refusal": float(g.loc[g["is_refusal"], "delta_cos"].mean()) if n_ref else float("nan"),
        })
    df_rel = pd.DataFrame(rel_rows)
    rel_path = os.path.join(args.outdir, "relation_summary.csv.gz")
    df_rel.to_csv(rel_path, index=False, compression="gzip")
    print(f"[write] {rel_path}")

    # ---------------------------
    # Correlations across relations (per model, per target)
    # ---------------------------
    corr_rows = []
    print("\n== per-model correlation across relations ==")
    for target in targets:
        _, _, _, denom_col, mean_col = subset_and_target(df, target)

        if target == "hall_given_value":
            rate_col = "hall_rate_given_value"
        elif target == "hall_given_noncorrect":
            rate_col = "hall_rate_given_noncorrect"
        else:
            rate_col = "refusal_rate"

        print(f"\n[target={target}] x={mean_col} y={rate_col} denom={denom_col} (min_rel_denom={args.min_rel_denom})")

        for mk, sub in df_rel.groupby("model_key"):
            x = sub[mean_col].to_numpy(dtype=float)
            y = sub[rate_col].to_numpy(dtype=float)
            denom = sub[denom_col].to_numpy(dtype=float)

            m_all = np.isfinite(x) & np.isfinite(y) & np.isfinite(denom) & (denom > 0)
            r_all = pearson_r(x[m_all], y[m_all])
            n_all = int(m_all.sum())

            if args.min_rel_denom > 0:
                m_stable = m_all & (denom >= args.min_rel_denom)
            else:
                m_stable = m_all
            r_stable = pearson_r(x[m_stable], y[m_stable])
            n_stable = int(m_stable.sum())

            r_w = weighted_pearson_r(x, y, denom)

            print(f"{mk}: r_all={r_all:.4f} (n={n_all})  r_stable={r_stable:.4f} (n={n_stable})  r_weighted={r_w:.4f}")

            corr_rows.append({
                "split_filter": args.split,
                "target": target,
                "model_key": mk,
                "x": mean_col,
                "y": rate_col,
                "denom_col": denom_col,
                "min_rel_denom": int(args.min_rel_denom),
                "n_rel_all": n_all,
                "pearson_r_all": float(r_all),
                "n_rel_stable": n_stable,
                "pearson_r_stable": float(r_stable),
                "pearson_r_weighted": float(r_w),
            })

    corr_path = os.path.join(args.outdir, "correlation_summary.csv")
    pd.DataFrame(corr_rows).to_csv(corr_path, index=False)
    print(f"\n[write] {corr_path}")

    # ---------------------------
    # Fits + plots (logistic only)
    # ---------------------------
    fit_rows = []
    print("\n[fit+plot] ...")
    for target in targets:
        plots_dir = os.path.join(args.outdir, "plots", target)
        os.makedirs(plots_dir, exist_ok=True)

        for mk in sorted(df["model_key"].unique()):
            df_m = df[df["model_key"] == mk].copy()

            # pooled plot
            pooled_pdf = os.path.join(plots_dir, f"pooled_{mk}.pdf")
            with PdfPages(pooled_pdf) as pdf:
                x, y, ylabel, _, _ = subset_and_target(df_m, target)
                fig, ax = plt.subplots(figsize=(7.6, 4.8))

                if len(x) >= 1:
                    bins = equal_count_bins(x, y, n_bins=max(8, args.n_bins * 2))
                    plot_bins_with_ci(ax, bins, label="binned (equal-count)")
                    x_min, x_max = float(np.min(x)), float(np.max(x))

                    if len(x) >= args.min_fit_n and np.unique(y).size >= 2:
                        fit = fit_logistic(x, y, seed=args.seed)
                        plot_logistic_curve(ax, fit["est"], x_min, x_max, "logistic (CV)")
                        slope = float(fit["logit_slope"])
                        cv_ll = float(fit["cv_logloss"])
                        ax.set_title(f"{mk} pooled (split={args.split}) target={target}\n"
                                     f"n={len(y)} base_rate={float(np.mean(y)):.3f} slope={slope:.3g} cv_logloss={cv_ll:.3f}")
                    else:
                        ax.set_title(f"{mk} pooled (split={args.split}) target={target}\n"
                                     f"n={len(y)} base_rate={float(np.mean(y)):.3f} (no fit: n<{args.min_fit_n} or one class)")
                    ax.set_xlim(x_min - 0.02, x_max + 0.02)
                else:
                    ax.text(0.5, 0.5, "No samples for this target subset", ha="center", va="center")

                ax.set_xlabel("Δcos (per-triple, gold object)")
                ax.set_ylabel(ylabel)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            print(f"[plot] {pooled_pdf}")

            # per-relation multipage PDF
            rel_pdf = os.path.join(plots_dir, f"relation_fits_{mk}.pdf")
            with PdfPages(rel_pdf) as pdf:
                rels = sorted(df_m["relation_key"].unique())[: args.max_relations_per_model]

                for rk in tqdm(rels, desc=f"{target}[{mk}] relations"):
                    g = df_m[df_m["relation_key"] == rk].copy()
                    rg = str(g["relation_group"].iloc[0]) if len(g) else ""
                    rn = str(g["relation_name"].iloc[0]) if len(g) else rk

                    n_total = int(len(g))
                    n_ref = int(g["is_refusal"].sum())
                    n_cor = int(g["is_correct"].sum())
                    n_hall = int(g["is_hall"].sum())
                    n_value = int(g["is_value"].sum())
                    n_noncorrect = int(g["is_noncorrect"].sum())

                    x, y, ylabel, denom_col, _ = subset_and_target(g, target)
                    denom_n = {
                        "n_total": n_total,
                        "n_value": n_value,
                        "n_noncorrect": n_noncorrect,
                    }[denom_col]

                    include_rel = True
                    if args.min_rel_denom > 0 and denom_n < args.min_rel_denom:
                        include_rel = False

                    row = {
                        "split_filter": args.split,
                        "target": target,
                        "model_key": mk,
                        "relation_key": rk,
                        "relation_group": rg,
                        "relation_name": rn,

                        "n_total": n_total,
                        "n_refusal": n_ref,
                        "n_correct": n_cor,
                        "n_hallucination": n_hall,
                        "n_value": n_value,
                        "n_noncorrect": n_noncorrect,

                        "subset_denom_col": denom_col,
                        "subset_n": int(denom_n),
                        "include_rel": bool(include_rel),
                        "delta_cos_mean_subset": float(np.mean(x)) if len(x) else float("nan"),
                        "base_rate_subset": float(np.mean(y)) if len(y) else float("nan"),

                        "logistic_slope": float("nan"),
                        "logistic_intercept": float("nan"),
                        "cv_folds": 0,
                        "cv_logloss": float("nan"),
                        "cv_auc": float("nan"),
                        "cv_brier": float("nan"),
                        "fit_ok": False,
                    }

                    if len(x) >= args.min_fit_n and np.unique(y).size >= 2:
                        fit = fit_logistic(x, y, seed=args.seed)
                        row.update({
                            "logistic_slope": _safe_float(fit.get("logit_slope")),
                            "logistic_intercept": _safe_float(fit.get("logit_intercept")),
                            "cv_folds": int(fit.get("cv_folds", 0)),
                            "cv_logloss": _safe_float(fit.get("cv_logloss")),
                            "cv_auc": _safe_float(fit.get("cv_auc")),
                            "cv_brier": _safe_float(fit.get("cv_brier")),
                            "fit_ok": True,
                        })

                    fit_rows.append(row)

                    if not include_rel:
                        continue  # skip plotting this relation

                    fig, ax = plt.subplots(figsize=(7.6, 4.8))

                    if len(x) >= 1:
                        bins = equal_count_bins(x, y, n_bins=args.n_bins)
                        plot_bins_with_ci(ax, bins, label="binned (equal-count)")
                        x_min, x_max = float(np.min(x)), float(np.max(x))
                        ax.set_xlim(x_min - 0.02, x_max + 0.02)

                        if row["fit_ok"]:
                            fit = fit_logistic(x, y, seed=args.seed)
                            plot_logistic_curve(ax, fit["est"], x_min, x_max, "logistic")
                    else:
                        ax.text(0.5, 0.5, "No samples for this target subset", ha="center", va="center")

                    ax.set_title(
                        f"{mk} | {rk} ({rn}) [{rg}] split={args.split} target={target}\n"
                        f"n_total={n_total} n_value={n_value} n_noncorrect={n_noncorrect} base_rate={row['base_rate_subset']:.3f}"
                    )
                    ax.set_xlabel("Δcos (per-triple, gold object)")
                    ax.set_ylabel(ylabel)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="best", fontsize=9)
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            print(f"[plot] {rel_pdf}")

    fit_path = os.path.join(args.outdir, "fit_summary.csv.gz")
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False, compression="gzip")
    print(f"[write] {fit_path}")

    print(f"\n[done] outdir={args.outdir}")


if __name__ == "__main__":
    main()
