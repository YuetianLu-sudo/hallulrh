#!/usr/bin/env python
import argparse
import csv
import math
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Optional

CANON_LABELS = ["REFUSAL", "HALLUCINATION", "OTHER"]


def normalize_label(text: Optional[str]) -> Optional[str]:
    """Map free-form judge labels to canonical labels.

    Returns one of {"REFUSAL", "HALLUCINATION", "OTHER"} or None.
    """
    if text is None:
        return None
    s = text.strip().upper()
    if not s:
        return None

    # Direct matches
    if s in CANON_LABELS:
        return s

    # Common variants
    if s in {"REFUSE", "REFUSAL/UNKNOWN", "UNKNOWN"}:
        return "REFUSAL"
    if s in {"HALLUCINATE", "HALLUCINATION/ANSWER"}:
        return "HALLUCINATION"
    if s in {"UNCLEAR", "BORDERLINE", "MIXED"}:
        return "OTHER"

    return None


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Compute p-hat and 95% Wilson interval."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n)
        )
        / denom
    )
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return phat, lower, upper


def detect_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    """Return first column name present in fieldnames from candidates."""
    s = set(fieldnames)
    for c in candidates:
        if c in s:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Summarize LM-as-judge labels per (model, task)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV with judge_label column (output of lm_judge_gemini.py judge-csv).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV file to write the summary table.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    task_col = detect_column(fieldnames, ["task", "relation", "property"])
    model_col = detect_column(fieldnames, ["model_name", "model"])
    judge_col = detect_column(fieldnames, ["judge_label", "gemini_label", "llm_label"])

    if judge_col is None:
        raise SystemExit(
            f"Could not find judge label column in {fieldnames}. "
            f"Expected one of ['judge_label', 'gemini_label', 'llm_label']."
        )

    # key: (model_name, task) -> counts
    groups: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for row in rows:
        model_name = row.get(model_col, "") if model_col is not None else ""
        task = row.get(task_col, "") if task_col is not None else ""
        raw = row.get(judge_col, "")
        lab = normalize_label(raw)

        if lab is None:
            continue
        groups[(model_name, task)][lab] += 1
        groups[(model_name, task)]["TOTAL"] += 1

    # Prepare summary rows
    summary_rows: List[Dict[str, str]] = []

    print("model_name,task,total,refusal,hallucination,other,"
          "refusal_rate,refusal_ci_low,refusal_ci_high,"
          "halluc_rate,halluc_ci_low,halluc_ci_high")

    for (model_name, task), cnt in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        total = cnt["TOTAL"]
        ref = cnt.get("REFUSAL", 0)
        hal = cnt.get("HALLUCINATION", 0)
        oth = cnt.get("OTHER", 0)

        # Wilson on all examples (including OTHER)
        ref_p, ref_lo, ref_hi = wilson_interval(ref, total)
        hal_p, hal_lo, hal_hi = wilson_interval(hal, total)

        print(
            f"{model_name},{task},{total},"
            f"{ref},{hal},{oth},"
            f"{ref_p:.3f},{ref_lo:.3f},{ref_hi:.3f},"
            f"{hal_p:.3f},{hal_lo:.3f},{hal_hi:.3f}"
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "task": task,
                "total": str(total),
                "refusal": str(ref),
                "hallucination": str(hal),
                "other": str(oth),
                "refusal_rate": f"{ref_p:.6f}",
                "refusal_ci_low": f"{ref_lo:.6f}",
                "refusal_ci_high": f"{ref_hi:.6f}",
                "halluc_rate": f"{hal_p:.6f}",
                "halluc_ci_low": f"{hal_lo:.6f}",
                "halluc_ci_high": f"{hal_hi:.6f}",
            }
        )

    if args.output is not None:
        out_fields = list(summary_rows[0].keys()) if summary_rows else []
        with open(args.output, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"[summary] Wrote summary CSV to {args.output}")


if __name__ == "__main__":
    main()
