#!/usr/bin/env python3
import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

MODEL_ORDER = [
    "gemma_7b_it",
    "llama3_1_8b_instruct",
    "mistral_7b_instruct",
    "qwen2_5_7b_instruct",
]

REL_ORDER = [
    "father",
    "instrument",
    "sport",
    "company_ceo",
    "company_hq",
    "country_language",
]


def _corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (pearson_r, pearson_p, spearman_rho, spearman_p)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to behavior_plus_lre.csv (one row per model×relation)")
    ap.add_argument("--x", default="cos_improvement", help="X column (default: cos_improvement)")
    ap.add_argument("--y", default="halluc_rate", help="Y column (default: halluc_rate)")
    ap.add_argument("--group", default="model_key", help="Group column (default: model_key)")
    ap.add_argument("--loo", action="store_true", help="Also run leave-one-relation-out per model")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {args.x, args.y, args.group, "relation"}
    missing = required - set(df.columns)
    if missing:
        print(f"[error] Missing columns: {sorted(missing)}", file=sys.stderr)
        print(f"[info] Available columns: {list(df.columns)}", file=sys.stderr)
        raise SystemExit(1)

    # Enforce stable ordering if possible
    if set(MODEL_ORDER).issubset(set(df[args.group].unique())):
        model_list = MODEL_ORDER
    else:
        model_list = sorted(df[args.group].unique().tolist())

    print("=== Per-model correlations (n = #relations per model) ===")
    rows = []
    for mk in model_list:
        sub = df[df[args.group] == mk].copy()
        # order relations if possible
        if set(REL_ORDER).issubset(set(sub["relation"].unique())):
            sub["relation"] = pd.Categorical(sub["relation"], categories=REL_ORDER, ordered=True)
            sub = sub.sort_values("relation")

        x = sub[args.x].to_numpy(dtype=float)
        y = sub[args.y].to_numpy(dtype=float)
        pr, pp, sr, sp = _corr(x, y)
        rows.append((mk, len(sub), pr, pp, sr, sp))

    out = pd.DataFrame(rows, columns=["model", "n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"])
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # pooled correlation over all points
    print("\n=== Pooled correlation over all model×relation points ===")
    pr, pp, sr, sp = _corr(df[args.x].to_numpy(dtype=float), df[args.y].to_numpy(dtype=float))
    print(f"Pooled Pearson r = {pr:.3f} (p={pp:.4f})")
    print(f"Pooled Spearman ρ = {sr:.3f} (p={sp:.4f})")

    # Leave-one-relation-out sensitivity per model
    if args.loo:
        print("\n=== Leave-one-relation-out (LOO) sensitivity per model ===")
        for mk in model_list:
            sub = df[df[args.group] == mk].copy()
            rels = sub["relation"].tolist()
            loo_rows = []
            for rel in rels:
                sub2 = sub[sub["relation"] != rel]
                x = sub2[args.x].to_numpy(dtype=float)
                y = sub2[args.y].to_numpy(dtype=float)
                pr2, _, sr2, _ = _corr(x, y)
                loo_rows.append((rel, pr2, sr2))
            loo_df = pd.DataFrame(loo_rows, columns=["dropped_relation", "pearson_r", "spearman_rho"])
            print(f"\nModel: {mk}")
            print(loo_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            print(
                f"  Pearson r range (LOO): [{loo_df['pearson_r'].min():.3f}, {loo_df['pearson_r'].max():.3f}]"
            )
            print(
                f"  Spearman ρ range (LOO): [{loo_df['spearman_rho'].min():.3f}, {loo_df['spearman_rho'].max():.3f}]"
            )


if __name__ == "__main__":
    main()
