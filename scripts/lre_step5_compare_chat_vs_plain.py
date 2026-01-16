#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare step5 outputs between:
  A) chat_template version (with --use_chat_template)
  B) plain concatenation version (no --use_chat_template)

Reads:
  <BASE>/<model_key>/relation_summary.csv.gz

Outputs:
  - merged CSV with per (model, relation) deltas
  - per-model Pearson r between delta_cos_mean_test (chat vs plain)
"""

import argparse
import gzip
import csv
from pathlib import Path
from typing import Dict, Tuple, List
import math

def read_gz_csv(path: Path) -> List[Dict[str, str]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def pearson(x: List[float], y: List[float]) -> float:
    pairs = [(a,b) for a,b in zip(x,y) if (not math.isnan(a) and not math.isnan(b))]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs)/len(xs)
    my = sum(ys)/len(ys)
    num = sum((a-mx)*(b-my) for a,b in pairs)
    denx = math.sqrt(sum((a-mx)**2 for a in xs))
    deny = math.sqrt(sum((b-my)**2 for b in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num/(denx*deny)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat_base", required=True, help="OUTBASE for chat_template run")
    ap.add_argument("--plain_base", required=True, help="OUTBASE for no-template run")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_keys", nargs="+", default=[
        "llama3_1_8b_instruct","gemma_7b_it","mistral_7b_instruct","qwen2_5_7b_instruct"
    ])
    args = ap.parse_args()

    chat_base = Path(args.chat_base)
    plain_base = Path(args.plain_base)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    merged_rows = []

    for mk in args.model_keys:
        chat_sum = chat_base / mk / "relation_summary.csv.gz"
        plain_sum = plain_base / mk / "relation_summary.csv.gz"
        if not chat_sum.exists():
            raise FileNotFoundError(f"missing: {chat_sum}")
        if not plain_sum.exists():
            raise FileNotFoundError(f"missing: {plain_sum}")

        chat_rows = read_gz_csv(chat_sum)
        plain_rows = read_gz_csv(plain_sum)

        # index by relation_key
        idx_chat = {r["relation_key"]: r for r in chat_rows}
        idx_plain = {r["relation_key"]: r for r in plain_rows}

        rel_keys = sorted(set(idx_chat.keys()) & set(idx_plain.keys()))
        xs, ys = [], []

        for rk in rel_keys:
            a = idx_chat[rk]
            b = idx_plain[rk]
            dc_chat = to_float(a.get("delta_cos_mean_test","nan"))
            dc_plain = to_float(b.get("delta_cos_mean_test","nan"))
            xs.append(dc_chat); ys.append(dc_plain)

            merged_rows.append({
                "model_key": mk,
                "relation_key": rk,
                "relation_group": a.get("relation_group",""),
                "relation_name": a.get("relation_name",""),
                "n_used_chat": a.get("n_used",""),
                "n_used_plain": b.get("n_used",""),
                "delta_cos_mean_test_chat": a.get("delta_cos_mean_test",""),
                "delta_cos_mean_test_plain": b.get("delta_cos_mean_test",""),
                "delta_diff_chat_minus_plain": ("" if (math.isnan(dc_chat) or math.isnan(dc_plain)) else f"{(dc_chat-dc_plain):.6f}"),
                "subject_layer_chat": a.get("subject_layer",""),
                "object_layer_chat": a.get("object_layer",""),
                "subject_layer_plain": b.get("subject_layer",""),
                "object_layer_plain": b.get("object_layer",""),
            })

        r = pearson(xs, ys)
        print(f"[compare] {mk}: Pearson r(delta_cos_mean_test chat vs plain) = {r:.4f} (n_rel={len(rel_keys)})")

    fieldnames = list(merged_rows[0].keys()) if merged_rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in merged_rows:
            w.writerow(row)

    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()
