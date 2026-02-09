#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp2: Linear core vs nonlinear periphery inside each LRE relation.

Goal:
  Avoid the triviality: "select top delta-cos triples -> subrelation has higher delta-cos".
  We do *out-of-sample* definition:
    - define core threshold on TRAIN (top-q delta per triple)
    - evaluate core/periphery deltas on TEST

Inputs:
  --deltacos-root: step5 deltacos directory, e.g.
      data/lre_hernandez/deltacos/lre_step5_deltacos_gold_YYYYMMDD_HHMMSS
  Under each model subdir, we expect some *example-level* csv.gz with relation_key and either:
      - delta column, or
      - (cos_pred - cos_base)

Outputs:
  outdir/core_periphery_summary.csv
  plus printed top relations by core-periphery gap per model.
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MODELS = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

# Candidate column names (robust to file schema differences)
REL_CANDS = ["relation_key", "relation", "rel", "predicate"]
SPLIT_CANDS = ["split", "partition", "set", "fold", "is_test", "is_train"]

DELTA_CANDS = [
    "delta_cos",
    "delta_cos_test",
    "delta_cos_value",
    "delta_cos_mean",
    "delta_cos_mean_test",
    "cos_improvement",
]
COS_CANDS = [
    "cos",
    "cos_test",
    "cos_mean",
    "cos_mean_test",
    "cos_lre",
    "lre_cos",
    "cos_pred",
    "cos_hat",
]
BASE_COS_CANDS = [
    "base_cos",
    "base_cos_test",
    "base_cos_mean",
    "base_cos_mean_test",
    "cos_base",
    "cos_baseline",
]

# Some step5 pipelines store per-example rows under names like "pairs.csv.gz", "triples.csv.gz", etc.
PRIORITY_PATTERNS = [
    "*example*.csv.gz",
    "*pair*.csv.gz",
    "*pairs*.csv.gz",
    "*triple*.csv.gz",
    "*triples*.csv.gz",
    "*metrics*.csv.gz",
]


def pick_col(cols: List[str], cands: List[str]) -> Optional[str]:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def _try_read_head(path: str, nrows: int = 10) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, compression="gzip", nrows=nrows)
    except Exception:
        return None


def find_example_level_file(model_dir: str) -> str:
    """
    Find a gz CSV that looks like per-example (not relation_summary).
    We screen candidates by checking required columns.
    """
    candidates: List[str] = []

    for pat in PRIORITY_PATTERNS:
        candidates.extend(glob.glob(os.path.join(model_dir, "**", pat), recursive=True))

    if not candidates:
        candidates = glob.glob(os.path.join(model_dir, "**", "*.csv.gz"), recursive=True)

    candidates = sorted(set(candidates))
    candidates = [
        c for c in candidates
        if os.path.basename(c) != "relation_summary.csv.gz"
        and "relation_summary" not in os.path.basename(c)
    ]

    for path in candidates:
        head = _try_read_head(path, nrows=10)
        if head is None or head.empty:
            continue
        cols = [str(c) for c in head.columns.tolist()]
        rel_col = pick_col(cols, REL_CANDS)
        if rel_col is None:
            continue

        delta_col = pick_col(cols, DELTA_CANDS)
        cos_col = pick_col(cols, COS_CANDS)
        base_col = pick_col(cols, BASE_COS_CANDS)

        if delta_col is not None or (cos_col is not None and base_col is not None):
            return path

    # If we get here, we couldn't find a usable file.
    preview = "\n  ".join(candidates[:25])
    raise FileNotFoundError(
        f"Could not find an example-level .csv.gz under: {model_dir}\n"
        f"Tried patterns={PRIORITY_PATTERNS}\n"
        f"Found {len(candidates)} candidates (showing up to 25):\n  {preview}\n\n"
        f"Likely cause: step5 only wrote relation_summary.csv.gz but not per-example rows.\n"
        f"Fix: re-run the deltacos pipeline with 'save per-example/pairs' enabled, or locate a step4 cache "
        f"that contains per-pair cos/base_cos."
    )


def normalize_split(series: pd.Series) -> Optional[np.ndarray]:
    """
    Map a split-like column to {'train','test'}.
    Accepts:
      - strings containing 'train'/'test'
      - boolean
      - 0/1 indicators (1=test)
    """
    if series is None:
        return None

    # bool
    if series.dtype == bool:
        return np.where(series.to_numpy(), "test", "train")

    # numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        vals = series.dropna().unique()
        if len(vals) <= 3 and set(map(int, vals)) <= {0, 1}:
            arr = series.fillna(0).astype(int).to_numpy()
            return np.where(arr == 1, "test", "train")

    s = series.astype(str).str.lower()
    if s.str.contains("train").any() or s.str.contains("test").any():
        return np.where(s.str.contains("test"), "test", "train")

    if s.isin(["train", "test"]).any():
        return np.where(s == "test", "test", "train")

    return None


def assign_random_split(df: pd.DataFrame, seed: int, test_frac: float) -> np.ndarray:
    """
    If the input file has no split column, create a per-relation random split.
    """
    rng = np.random.default_rng(seed)
    split = np.full(len(df), "train", dtype=object)

    # Split within each relation to keep relation sizes stable
    for rel, idxs in df.groupby("relation_key").indices.items():
        idx = np.array(list(idxs), dtype=int)
        n = len(idx)
        if n <= 1:
            continue
        n_test = max(1, int(round(test_frac * n)))
        test_idx = rng.choice(idx, size=n_test, replace=False)
        split[test_idx] = "test"

    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltacos-root", required=True, help="Step5 deltacos directory root.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--core-q", type=float, default=0.30, help="Core fraction defined on TRAIN (top-q delta). Default=0.30")
    ap.add_argument("--seed", type=int, default=0, help="Seed for random split fallback.")
    ap.add_argument("--test-frac", type=float, default=0.25, help="Test fraction if split missing. Default=0.25")
    ap.add_argument("--min-total", type=int, default=30, help="Minimum total examples per relation.")
    ap.add_argument("--min-train", type=int, default=10, help="Minimum train examples per relation.")
    ap.add_argument("--min-test", type=int, default=5, help="Minimum test examples per relation.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for mk in MODELS:
        model_dir = os.path.join(args.deltacos_root, mk)
        if not os.path.isdir(model_dir):
            print(f"[warn] missing model dir: {model_dir}", file=sys.stderr)
            continue

        ex_path = find_example_level_file(model_dir)
        print(f"[info] {mk}: using example-level file: {ex_path}", file=sys.stderr)

        df = pd.read_csv(ex_path, compression="gzip")

        cols = [str(c) for c in df.columns.tolist()]
        rel_col = pick_col(cols, REL_CANDS)
        if rel_col is None:
            raise RuntimeError(f"{mk}: cannot find relation column in {ex_path}. cols={cols}")

        df["relation_key"] = df[rel_col].astype(str)

        split_col = pick_col(cols, SPLIT_CANDS)
        split = None
        if split_col is not None:
            split = normalize_split(df[split_col])

        if split is None:
            # Fallback: random split within each relation
            split = assign_random_split(df[["relation_key"]].copy(), seed=args.seed, test_frac=args.test_frac)

        df["_split"] = split

        # delta computation
        delta_col = pick_col(cols, DELTA_CANDS)
        if delta_col is not None:
            df["_delta"] = pd.to_numeric(df[delta_col], errors="coerce")
        else:
            cos_col = pick_col(cols, COS_CANDS)
            base_col = pick_col(cols, BASE_COS_CANDS)
            if cos_col is None or base_col is None:
                raise RuntimeError(
                    f"{mk}: cannot compute delta; need delta col or (cos, base_cos) cols.\n"
                    f"file={ex_path}\ncols={cols}"
                )
            df["_delta"] = pd.to_numeric(df[cos_col], errors="coerce") - pd.to_numeric(df[base_col], errors="coerce")

        df = df[np.isfinite(df["_delta"].to_numpy(float))].copy()

        # per relation
        for rel, g in df.groupby("relation_key"):
            n_total = int(len(g))
            if n_total < args.min_total:
                continue

            g_train = g[g["_split"] == "train"]
            g_test = g[g["_split"] == "test"]
            n_train = int(len(g_train))
            n_test = int(len(g_test))
            if n_train < args.min_train or n_test < args.min_test:
                continue

            train_delta = g_train["_delta"].to_numpy(float)
            test_delta = g_test["_delta"].to_numpy(float)

            # Core threshold defined on TRAIN
            thr = float(np.nanquantile(train_delta, 1.0 - args.core_q))

            core_mask = test_delta >= thr
            n_core = int(core_mask.sum())
            n_peri = int((~core_mask).sum())

            mean_all = float(np.nanmean(test_delta))
            mean_core = float(np.nanmean(test_delta[core_mask])) if n_core > 0 else float("nan")
            mean_peri = float(np.nanmean(test_delta[~core_mask])) if n_peri > 0 else float("nan")

            # Concentration on positive deltas (avoid negative-sum weirdness)
            pos = np.maximum(test_delta, 0.0)
            pos_sum = float(pos.sum())
            pos_share_core = float(np.maximum(test_delta[core_mask], 0.0).sum() / pos_sum) if pos_sum > 0 else float("nan")

            rows.append({
                "model_key": mk,
                "relation_key": rel,
                "n_total": n_total,
                "n_train": n_train,
                "n_test": n_test,
                "core_q": float(args.core_q),
                "thr_train_delta": thr,
                "delta_mean_test": mean_all,
                "delta_mean_test_core": mean_core,
                "delta_mean_test_periphery": mean_peri,
                "core_frac_test": float(n_core / max(1, n_test)),
                "gap_core_minus_periphery": float(mean_core - mean_peri) if np.isfinite(mean_core) and np.isfinite(mean_peri) else float("nan"),
                "gap_core_minus_all": float(mean_core - mean_all) if np.isfinite(mean_core) else float("nan"),
                "pos_share_core": pos_share_core,
            })

    if not rows:
        raise RuntimeError("No rows produced. Most likely: per-example files missing, or filters too strict.")

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.outdir, "core_periphery_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"[done] wrote: {out_path}")

    # Print top gaps per model
    for mk in MODELS:
        sub = out[out["model_key"] == mk].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("gap_core_minus_periphery", ascending=False)
        top = sub.head(8)[["relation_key","n_test","delta_mean_test","delta_mean_test_core","delta_mean_test_periphery","gap_core_minus_periphery","pos_share_core"]]
        print(f"\n== {mk}: top relations by core-periphery gap (defined on TRAIN, evaluated on TEST) ==")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
