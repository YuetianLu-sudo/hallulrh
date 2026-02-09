#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a compact markdown sanity report for core/periphery samples.

Expected input CSV filename pattern:
  <model_key>.<relation_key>.core_periphery_<split>.csv

Columns expected (at least):
  bucket, delta_cos, subject, gold_object
Optional:
  model_key, relation_key

Outputs:
  - sanity_report_<relation>_<split>.md
  - sanity_summary_<relation>_<split>.csv
"""

import argparse
import glob
import os
import re
import pandas as pd


def last_token(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    parts = re.split(r"\s+", s)
    return parts[-1] if parts else ""


def surname_match_rate(df: pd.DataFrame) -> float:
    # English comment: heuristic "surname" = last whitespace-delimited token
    if df.empty:
        return float("nan")
    subj_last = df["subject"].astype(str).map(last_token)
    obj_last = df["gold_object"].astype(str).map(last_token)
    m = (subj_last != "") & (obj_last != "")
    if m.sum() == 0:
        return float("nan")
    return float((subj_last[m] == obj_last[m]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-dir", required=True, help="Dir containing sampled CSVs")
    ap.add_argument("--relation", required=True, help="relation_key, e.g., product_by_company or person_father")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--show-n", type=int, default=12, help="How many examples to show per bucket")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pattern = os.path.join(args.sanity_dir, f"*.{args.relation}.core_periphery_{args.split}.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"[error] no CSV matched: {pattern}")

    md = []
    sum_rows = []

    for p in paths:
        df = pd.read_csv(p)

        mk = None
        if "model_key" in df.columns and df["model_key"].dropna().size > 0:
            mk = str(df["model_key"].dropna().iloc[0])
        if not mk:
            mk = os.path.basename(p).split(".")[0]

        need = {"bucket", "delta_cos", "subject", "gold_object"}
        miss = need - set(df.columns)
        if miss:
            raise SystemExit(f"[error] {os.path.basename(p)} missing columns: {sorted(miss)}")

        md.append(f"# {args.relation} — core/periphery sanity ({args.split})")
        md.append("")
        md.append(f"## {mk}")
        md.append("")

        for bucket in ["core_top", "periphery_bottom"]:
            sub = df[df["bucket"] == bucket].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("delta_cos", ascending=(bucket != "core_top"))
            top_objs = sub["gold_object"].value_counts().head(8)

            sm = surname_match_rate(sub)
            sm_str = "n/a" if pd.isna(sm) else f"{sm:.3f}"

            sum_rows.append({
                "model_key": mk,
                "relation_key": args.relation,
                "bucket": bucket,
                "n": int(len(sub)),
                "top_objects": "; ".join([f"{k}({int(v)})" for k, v in top_objs.items()]),
                "surname_match_rate": sm,
            })

            md.append(f"### {bucket} (n={len(sub)})")
            md.append("Top objects: " + ", ".join([f"{k} ({int(v)})" for k, v in top_objs.items()]))
            md.append(f"Surname-match rate (heuristic): {sm_str}")
            md.append("")
            md.append("| Δcos | subject | gold_object |")
            md.append("|---:|---|---|")

            show = sub.head(args.show_n)
            for _, r in show.iterrows():
                md.append(f"| {float(r['delta_cos']):.3f} | {str(r['subject'])} | {str(r['gold_object'])} |")

            md.append("")

    out_md = os.path.join(args.outdir, f"sanity_report_{args.relation}_{args.split}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md).strip() + "\n")
    print("[done] wrote:", out_md)

    out_csv = os.path.join(args.outdir, f"sanity_summary_{args.relation}_{args.split}.csv")
    pd.DataFrame(sum_rows).to_csv(out_csv, index=False)
    print("[done] wrote:", out_csv)


if __name__ == "__main__":
    main()
