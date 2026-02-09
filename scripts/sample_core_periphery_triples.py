#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import pandas as pd

def open_gz_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")

def load_prompts_meta(prompts_jsonl: str, relation_key: str):
    """
    Returns dict: example_id(str) -> dict(subject, gold_object, prompt)
    """
    meta = {}
    with open(prompts_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("relation_key") != relation_key:
                continue
            ex_id = rec.get("id", rec.get("example_id"))
            if ex_id is None:
                continue
            ex_id = str(ex_id).strip()
            if not ex_id:
                continue
            meta[ex_id] = {
                "subject": rec.get("subject", ""),
                "gold_object": rec.get("gold_object", rec.get("gold_answer", "")),
                "prompt": rec.get("prompt", rec.get("question", "")),
            }
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="core_periphery_summary.csv")
    ap.add_argument("--deltacos-root", required=True, help=".../lre_step5_deltacos_gold_*/")
    ap.add_argument("--prompts", required=True, help="lre_prompts_qonly.jsonl")
    ap.add_argument("--relation", required=True, help="e.g., product_by_company")
    ap.add_argument("--model-key", required=True, help="e.g., gemma_7b_it")
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    ap.add_argument("--n-core", type=int, default=25)
    ap.add_argument("--n-periphery", type=int, default=25)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    summ = pd.read_csv(args.summary)
    row = summ[(summ["model_key"].astype(str) == args.model_key) &
               (summ["relation_key"].astype(str) == args.relation)]
    if row.empty:
        raise RuntimeError(f"No summary row found for model={args.model_key}, relation={args.relation}")

    thr = float(row["thr_train_delta"].iloc[0])
    core_q = float(row["core_q"].iloc[0])

    # Per-relation per-triple file from step5:
    per_rel = os.path.join(args.deltacos_root, args.model_key, "per_relation", f"{args.relation}.csv.gz")
    if not os.path.exists(per_rel):
        raise FileNotFoundError(f"Missing per_relation file: {per_rel}")

    df = open_gz_csv(per_rel)

    # Normalize id column name
    if "id" in df.columns:
        df["example_id"] = df["id"].astype(str)
    elif "example_id" in df.columns:
        df["example_id"] = df["example_id"].astype(str)
    else:
        raise RuntimeError(f"Cannot find id/example_id column in {per_rel}. cols={list(df.columns)}")

    if "delta_cos" not in df.columns:
        raise RuntimeError(f"Missing delta_cos in {per_rel}. cols={list(df.columns)}")

    if args.split != "all":
        if "split" not in df.columns:
            raise RuntimeError(f"Missing split column in {per_rel}. cols={list(df.columns)}")
        df = df[df["split"].astype(str) == args.split].copy()

    df["delta_cos"] = df["delta_cos"].astype(float)
    df["is_core"] = df["delta_cos"] >= thr

    n_total = len(df)
    n_core = int(df["is_core"].sum())
    n_per = n_total - n_core
    print(f"[info] {args.model_key} {args.relation} split={args.split}: n_total={n_total} core={n_core} periphery={n_per} thr_train_delta={thr:.6f} (core_q={core_q})")

    meta = load_prompts_meta(args.prompts, args.relation)

    # Join meta
    df["subject"] = df["example_id"].map(lambda x: meta.get(x, {}).get("subject", ""))
    df["gold_object"] = df["example_id"].map(lambda x: meta.get(x, {}).get("gold_object", ""))
    df["prompt"] = df["example_id"].map(lambda x: meta.get(x, {}).get("prompt", ""))

    # Pick extremes for interpretability:
    core = df[df["is_core"]].sort_values("delta_cos", ascending=False).head(args.n_core).copy()
    per  = df[~df["is_core"]].sort_values("delta_cos", ascending=True).head(args.n_periphery).copy()

    core["bucket"] = "core_top"
    per["bucket"]  = "periphery_bottom"

    out = pd.concat([core, per], ignore_index=True)
    if "model_key" in out.columns:
        out["model_key"] = args.model_key
    else:
        # patched: tolerate existing "model_key" column
        if "model_key" in out.columns:
            out["model_key"] = args.model_key
            out.insert(0, "model_key", out.pop("model_key"))
        else:
            out.insert(0, "model_key", args.model_key)
    # patched: tolerate existing "relation_key" column
    if "relation_key" in out.columns:
        out["relation_key"] = args.relation
        out.insert(1, "relation_key", out.pop("relation_key"))
    else:
        out.insert(1, "relation_key", args.relation)
    keep = ["model_key","relation_key","bucket","example_id","delta_cos","is_core","subject","gold_object","prompt"]
    out = out[keep]

    out_path = os.path.join(args.outdir, f"{args.model_key}.{args.relation}.core_periphery_{args.split}.csv")
    out.to_csv(out_path, index=False)
    print("[done] wrote:", out_path)

    # Print a few lines for quick eyeballing
    print("\n[peek] core_top (first 5):")
    print(core[["delta_cos","subject","gold_object"]].head(5).to_string(index=False))
    print("\n[peek] periphery_bottom (first 5):")
    print(per[["delta_cos","subject","gold_object"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
