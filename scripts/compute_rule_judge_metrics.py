#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute agreement between a deterministic regex-based judge and Gemini judge labels.

Expected input files: CSVs that contain at least:
  - answer: the model output string
  - judge_label: Gemini label in {"REFUSAL","HALLUCINATION"} (case-insensitive)

Example:
  python compute_rule_judge_metrics.py --glob "./data/judge_inputs/**/*with_judge*.csv"

This script is intentionally dependency-light (pandas only).
"""

import argparse
import glob
import io
import os
import re
import sys
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd


LABELS = ["REFUSAL", "HALLUCINATION"]


def normalize_text(s: str) -> str:
    """Normalize unicode punctuation to make regex matching stable."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    # Normalize common apostrophes/quotes/dashes
    s = (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("`", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("–", "-")
         .replace("—", "-")
    )
    return s


# A compact but high-coverage set of refusal templates.
# (Edit/extend these patterns if you want to match your paper's exact baseline.)
REFUSAL_PATTERNS: List[str] = [
    r"\bi (do not|don't) know\b",
    r"\bi (do not|don't) have\b.*\binformation\b",
    r"\bi (do not|don't) have\b.*\bdata\b",
    r"\bi (do not|don't) have\b.*\baccess\b",
    r"\bi (cannot|can't|could not|couldn't|unable to)\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bcan't\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bcannot\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bnot enough (information|context)\b",
    r"\binsufficient (information|context)\b",
    r"\bwithout (additional|more) information\b",
    r"\bno (reliable )?(information|data)\b",
    r"\bno\b.*\bavailable\b",
    r"\bnot\b.*\bavailable\b",
    r"\bnot provided\b",
    r"\bnot specified\b",
    r"\bdoes not specify\b",
    r"\bnot given\b",
    r"\bunknown\b",
    r"\bnot known\b",
    r"\bunsure\b",
    r"\bnot sure\b",
    r"\bi am not sure\b",
    r"\bi'm not sure\b",
    r"\bi am unsure\b",
    r"\bi'm unsure\b",
    r"\bi am not aware\b",
    r"\bi'm not aware\b",
    r"\bnot aware of\b",
    r"\bthe text (provided )?(does not|doesn't) (specify|reveal|provide|mention|state)\b",
    r"\bthe information (provided )?(does not|doesn't) (specify|reveal|provide|mention|state)\b",
    r"\bthe question (cannot|can't) be answered\b",
    r"\bcannot be answered\b",
    r"\bcan't be answered\b",
    r"\bno official\b.*\bspecified\b",
    r"\bnot (able|authorized) to\b",
    r"\bprivacy\b",
    r"\bsensitive\b",
    r"\bas an ai\b",
    r"\bi apologize\b",
    r"\bi'm sorry\b",
    r"\bi am sorry\b",
    r"\bfictional\b",
    r"\bdoes not exist\b",
    r"\bcan't browse\b",
    r"\bcannot browse\b",
    r"\bno access to\b.*\b(internet|web|database)\b",
]

REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in REFUSAL_PATTERNS), flags=re.IGNORECASE)


def rule_label(answer: str) -> str:
    a = normalize_text(answer).strip()
    if a == "":
        return "REFUSAL"
    # Common placeholders
    if re.fullmatch(r"(?i)(n/?a|none|null|unknown|not applicable)\.?", a):
        return "REFUSAL"
    # Template matching
    if REFUSAL_RE.search(a):
        return "REFUSAL"
    # Generic meta answers
    if re.fullmatch(r"(?i)(i am (an )?ai language model\.?|i am a helpful assistant\.?)", a):
        return "REFUSAL"
    return "HALLUCINATION"


def confusion_counts(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in labels or p not in labels:
            continue
        cm[t][p] += 1
    return cm


def accuracy_from_cm(cm: Dict[str, Dict[str, int]], labels: List[str]) -> float:
    total = sum(cm[t][p] for t in labels for p in labels)
    correct = sum(cm[l][l] for l in labels)
    return correct / total if total else float("nan")


def kappa_from_cm(cm: Dict[str, Dict[str, int]], labels: List[str]) -> float:
    """
    Cohen's kappa for 2 labels:
      kappa = (po - pe) / (1 - pe)
    where po is observed agreement, pe is expected agreement by chance.
    """
    total = sum(cm[t][p] for t in labels for p in labels)
    if total == 0:
        return float("nan")
    po = accuracy_from_cm(cm, labels)

    row_marg = {t: sum(cm[t][p] for p in labels) for t in labels}
    col_marg = {p: sum(cm[t][p] for t in labels) for p in labels}
    pe = sum(row_marg[l] * col_marg[l] for l in labels) / (total * total)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def precision_recall_from_cm(cm: Dict[str, Dict[str, int]], labels: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[t][l] for t in labels if t != l)
        fn = sum(cm[l][p] for p in labels if p != l)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        out[l] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
    return out


def read_csv_safely(path: str) -> pd.DataFrame:
    # Robust reading even if there are odd unicode bytes.
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help='Glob for input CSVs, e.g. "./data/judge_inputs/**/*with_judge*.csv"')
    ap.add_argument("--show_by_model", action="store_true", help="Also print metrics broken down by model_name")
    ap.add_argument("--show_by_task", action="store_true", help="Also print metrics broken down by task")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob, recursive=True))
    paths = [p for p in paths if p.lower().endswith(".csv") and "__macosx" not in p.lower()]
    if not paths:
        print(f"[ERROR] No CSV files matched: {args.glob}", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for p in paths:
        df = read_csv_safely(p)
        df["__source_file"] = p
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Basic validation
    required_cols = {"answer", "judge_label"}
    if not required_cols.issubset(set(df_all.columns)):
        missing = required_cols - set(df_all.columns)
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df_all.columns)}")

    y_true = [str(x).strip().upper() for x in df_all["judge_label"].tolist()]
    y_pred = [rule_label(x) for x in df_all["answer"].tolist()]

    cm = confusion_counts(y_true, y_pred, LABELS)
    acc = accuracy_from_cm(cm, LABELS)
    kappa = kappa_from_cm(cm, LABELS)
    pr = precision_recall_from_cm(cm, LABELS)

    print("== Overall (all files) ==")
    print(f"N = {len(y_true)}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Cohen's kappa = {kappa:.4f}")
    print("Confusion (rows=true, cols=pred):")
    print(f"           pred=REFUSAL   pred=HALLUCINATION")
    print(f"true=REFUSAL        {cm['REFUSAL']['REFUSAL']:6d}         {cm['REFUSAL']['HALLUCINATION']:6d}")
    print(f"true=HALLUCINATION  {cm['HALLUCINATION']['REFUSAL']:6d}         {cm['HALLUCINATION']['HALLUCINATION']:6d}")
    print()
    for l in LABELS:
        print(f"[{l}] precision={pr[l]['precision']:.4f}  recall={pr[l]['recall']:.4f}  tp={pr[l]['tp']} fp={pr[l]['fp']} fn={pr[l]['fn']}")

    def compute_on_subset(df_sub: pd.DataFrame, name: str) -> None:
        yt = [str(x).strip().upper() for x in df_sub["judge_label"].tolist()]
        yp = [rule_label(x) for x in df_sub["answer"].tolist()]
        cm_s = confusion_counts(yt, yp, LABELS)
        acc_s = accuracy_from_cm(cm_s, LABELS)
        k_s = kappa_from_cm(cm_s, LABELS)
        print(f"\n== {name} ==")
        print(f"N = {len(yt)}  Acc={acc_s:.4f}  Kappa={k_s:.4f}")

    if args.show_by_model and "model_name" in df_all.columns:
        for m in sorted(df_all["model_name"].dropna().unique()):
            compute_on_subset(df_all[df_all["model_name"] == m], f"By model: {m}")

    if args.show_by_task and "task" in df_all.columns:
        for t in sorted(df_all["task"].dropna().unique()):
            compute_on_subset(df_all[df_all["task"] == t], f"By task: {t}")

    # Convenience: print a LaTeX-ready 2-row table for precision/recall
    print("\n== LaTeX snippet for Table (precision/recall) ==")
    print(r"\textsc{Refusal} & %.3f & %.3f \\" % (pr["REFUSAL"]["precision"], pr["REFUSAL"]["recall"]))
    print(r"\textsc{Hallucination} & %.3f & %.3f \\" % (pr["HALLUCINATION"]["precision"], pr["HALLUCINATION"]["recall"]))
    print("\n(If you change REFUSAL_PATTERNS, re-run and update the paper numbers accordingly.)")


if __name__ == "__main__":
    main()
