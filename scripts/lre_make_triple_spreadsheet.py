#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge Step4 3-way labels with Step5 per-triple Δcos outputs, and write a spreadsheet.

Example:
  python scripts/lre_make_triple_spreadsheet.py \
    --labels_csv data/lre_hernandez/labeled/lre_qonly_20260104_231919_3way/labels.csv.gz \
    --step5_dir  data/lre_hernandez/deltacos/lre_step5_deltacos_gold_20260105_030210 \
    --out_csv_gz data/lre_hernandez/spreadsheets/lre_triples_qonly_gold_merged.csv.gz \
    --out_xlsx   data/lre_hernandez/spreadsheets/lre_triples_qonly_gold_merged.xlsx
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List

import pandas as pd


# Excel / openpyxl 不允许的控制字符（会导致写出来的 xlsx 损坏）
_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _clean_excel_cell(x):
    if pd.isna(x):
        return x
    if not isinstance(x, str):
        return x
    return _ILLEGAL_XLSX_RE.sub("", x)


def _require_cols(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[error] Missing columns in {where}: {missing}\nFound: {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, help="Step4 labels.csv.gz (3-way labels)")
    ap.add_argument("--step5_dir", required=True, help="Step5 output dir containing <model_key>/per_triple.csv.gz")
    ap.add_argument("--out_csv_gz", required=True, help="Output merged CSV.GZ path")
    ap.add_argument("--out_xlsx", default=None, help="Optional output XLSX path")
    ap.add_argument("--only_split", choices=["all", "train", "test"], default="all",
                    help="Filter by Step5 split column after merge (recommended: test)")
    ap.add_argument("--keep_full_answer", action="store_true",
                    help="Keep the full 'answer' column in outputs (default: drop it, keep answer_short)")
    ap.add_argument("--allow_model_id_mismatch", action="store_true",
                    help="Allow model_id mismatch between step4 and step5 (NOT recommended).")
    args = ap.parse_args()

    labels_path = Path(args.labels_csv)
    step5_dir = Path(args.step5_dir)
    out_csv = Path(args.out_csv_gz)
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else None

    print(f"[load] labels: {labels_path}")
    labels = pd.read_csv(labels_path)

    _require_cols(
        labels,
        cols=[
            "id", "model_key", "model_id",
            "relation_key", "relation_group", "relation_name",
            "subject", "gold_object",
            "prompt", "prompt_style",
            "answer_short",
            "label_3way",
        ],
        where="labels.csv.gz",
    )
    if "answer" not in labels.columns:
        # 有些版本可能没保留 answer，只要 answer_short 在就可以
        labels["answer"] = ""

    model_keys = sorted(labels["model_key"].dropna().unique().tolist())
    print(f"[info] found model_keys in labels: {model_keys}")

    merged_parts = []

    step5_required = [
        "id", "model_key", "model_id",
        "relation_key", "relation_group", "relation_name",
        "split", "base_cos", "cos", "delta_cos",
        "direction_norm", "subject_layer", "object_layer",
    ]

    for mk in model_keys:
        lab = labels[labels["model_key"] == mk].copy()
        per_triple_path = step5_dir / mk / "per_triple.csv.gz"
        if not per_triple_path.exists():
            raise FileNotFoundError(f"[error] Missing Step5 per_triple for {mk}: {per_triple_path}")

        print(f"[load] step5 per_triple: {per_triple_path}")
        pt = pd.read_csv(per_triple_path)
        _require_cols(pt, step5_required, where=str(per_triple_path))

        # ---- model_id 一致性检查（非常关键，避免你以后又 roll-back 混版本）----
        mid_lab = lab["model_id"].dropna().unique().tolist()
        mid_pt = pt["model_id"].dropna().unique().tolist()

        mid_lab_1 = mid_lab[0] if len(mid_lab) >= 1 else None
        mid_pt_1 = mid_pt[0] if len(mid_pt) >= 1 else None

        if mid_lab_1 != mid_pt_1:
            msg = (f"[error] model_id mismatch for model_key={mk}\n"
                   f"  step4 labels model_id = {mid_lab_1}\n"
                   f"  step5 deltacos model_id= {mid_pt_1}\n"
                   f"Fix: re-run step5 for this model with the SAME HF checkpoint as step3/step4.\n")
            if not args.allow_model_id_mismatch:
                print(msg, file=sys.stderr)
                sys.exit(2)
            else:
                print("[warn] " + msg)

        # ---- merge ----
        key_cols = ["model_key", "id", "relation_key", "relation_group", "relation_name", "model_id"]
        m = lab.merge(
            pt,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )

        # 必须全部 merge 上
        if m["delta_cos"].isna().any():
            n_na = int(m["delta_cos"].isna().sum())
            raise ValueError(f"[error] Merge produced {n_na} NA delta_cos for model_key={mk}. Keys likely mismatched.")

        merged_parts.append(m)
        print(f"[ok] merged {mk}: {len(m)} rows")

    merged = pd.concat(merged_parts, ignore_index=True)
    print(f"[info] merged total rows = {len(merged)}")

    # split 过滤（推荐只用 test 做后续分析，避免方向向量训练/测试泄漏）
    if args.only_split != "all":
        merged = merged[merged["split"] == args.only_split].copy()
        print(f"[info] after only_split={args.only_split}: rows={len(merged)}")

    # 默认不保留 full answer（太长且不利于表格查看）；但保留 answer_short
    if not args.keep_full_answer and "answer" in merged.columns:
        merged = merged.drop(columns=["answer"])

    # 排列一个更“读得懂”的列顺序
    preferred = [
        "model_key", "model_id",
        "id",
        "relation_key", "relation_group", "relation_name",
        "subject", "gold_object",
        "prompt_style", "prompt",
        "answer_short",
        "label_3way",
        "split",
        "subject_layer", "object_layer",
        "base_cos", "cos", "delta_cos",
        "direction_norm",
    ]
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    merged = merged[cols]

    # ---- write csv.gz ----
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] csv.gz -> {out_csv}")
    merged.to_csv(out_csv, index=False, compression="gzip")

    # ---- write xlsx (optional) ----
    if out_xlsx is not None:
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        print(f"[save] xlsx  -> {out_xlsx}")

        xdf = merged.copy()
        # 清理 Excel 不允许字符，避免写坏文件
        for col in xdf.select_dtypes(include=["object"]).columns:
            xdf[col] = xdf[col].map(_clean_excel_cell)

        # 直接写一个 sheet
        xdf.to_excel(out_xlsx, index=False)

    print("[done]")


if __name__ == "__main__":
    main()
