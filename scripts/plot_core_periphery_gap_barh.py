#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="core_periphery_summary.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-n-test", type=int, default=30)
    ap.add_argument("--topk", type=int, default=0,
                    help="Top-K by mean gap (0 means: plot ALL after filtering)")
    ap.add_argument("--metric", default="gap_core_minus_periphery")
    ap.add_argument("--require-all-models", action="store_true",
                    help="Keep only relations that appear for all models after filtering")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = {"model_key", "relation_key", "n_test", args.metric}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    df = df[df["model_key"].isin(MODEL_ORDER)].copy()
    df = df[df["n_test"].astype(float) >= float(args.min_n_test)].copy()
    df = df[np.isfinite(df[args.metric].astype(float))].copy()

    # Relation-level filter: min n_test across models
    min_nt = df.groupby("relation_key")["n_test"].min()
    keep_rels = set(min_nt[min_nt >= args.min_n_test].index.tolist())
    df = df[df["relation_key"].isin(keep_rels)].copy()

    if args.require_all_models:
        cnt = df.groupby("relation_key")["model_key"].nunique()
        keep_rels2 = set(cnt[cnt == len(MODEL_ORDER)].index.tolist())
        df = df[df["relation_key"].isin(keep_rels2)].copy()

    if df.empty:
        raise RuntimeError("No rows after filtering; lower --min-n-test or disable --require-all-models.")

    pv = df.pivot_table(
        index="relation_key",
        columns="model_key",
        values=args.metric,
        aggfunc="mean",
    )

    # Ensure column order
    pv = pv[[c for c in MODEL_ORDER if c in pv.columns]].copy()

    # Sort by cross-model mean gap (descending)
    pv["__mean__"] = pv.mean(axis=1)
    pv = pv.sort_values("__mean__", ascending=False)

    if args.topk and args.topk > 0:
        pv = pv.head(args.topk)

    pv = pv.drop(columns=["__mean__"])

    # Rename model columns for nicer legend
    pv = pv.rename(columns={k: MODEL_TITLE.get(k, k) for k in pv.columns})

    n_rel = len(pv)
    # Dynamic figure height
    fig_h = max(4.8, 0.38 * n_rel + 1.5)

    fig, ax = plt.subplots(figsize=(10.0, fig_h))

    pv.plot(kind="barh", ax=ax, width=0.85)

    ax.set_xlabel("Δcos gap (core − periphery) on TEST", fontsize=12)
    ax.set_ylabel("relation_key", fontsize=12)

    title = f"Core–periphery heterogeneity (min_n_test≥{args.min_n_test})"
    ax.set_title(title, fontsize=13)

    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Put largest at top
    ax.invert_yaxis()

    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()

    tag = f"min{args.min_n_test}"
    if args.topk and args.topk > 0:
        tag += f"_top{args.topk}"
    out_base = os.path.join(args.outdir, f"core_periphery_gap_barh_{tag}")

    fig.savefig(out_base + ".png", dpi=args.dpi)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)

    print("[done] wrote:", out_base + ".png")
    print("[done] wrote:", out_base + ".pdf")
    print("[info] n_relations =", n_rel)

if __name__ == "__main__":
    main()
