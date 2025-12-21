import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def normalize_label(x) -> str:
    """Map raw labels to {REFUSAL, HALLUCINATION, OTHER} robustly."""
    s = str(x).strip().upper()
    if s in {"REFUSAL", "HALLUCINATION", "OTHER"}:
        return s
    # Anything unexpected is treated as OTHER
    return "OTHER"


def detect_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first column name that exists in df, else raise."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of {candidates} in columns: {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Judged CSV (with judge_label column).")
    parser.add_argument("--output", type=str, required=True, help="Where to write summary CSV.")
    parser.add_argument(
        "--collapse-other-to-refusal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, treat OTHER as REFUSAL (binary: hallucination vs non-hallucination). Default: true.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Detect key columns (robust to small variations)
    label_col = detect_column(df, ["judge_label", "label", "pred_label"])
    if "task" in df.columns:
        task_col = "task"
    elif "relation" in df.columns:
        task_col = "relation"
    else:
        df["task"] = "all"
        task_col = "task"

    if "model_name" in df.columns:
        model_col = "model_name"
    else:
        df["model_name"] = in_path.stem
        model_col = "model_name"

    # Normalize labels
    df[label_col] = df[label_col].apply(normalize_label)

    rows_out: List[Dict] = []

    # Group by (model_name, task)
    grouped = df.groupby([model_col, task_col], dropna=False)
    for (model_name, task), g in grouped:
        total = int(len(g))

        refusal_raw = int((g[label_col] == "REFUSAL").sum())
        halluc = int((g[label_col] == "HALLUCINATION").sum())
        other_raw = int((g[label_col] == "OTHER").sum())

        if args.collapse_other_to_refusal:
            refusal = refusal_raw + other_raw
            other = 0
        else:
            refusal = refusal_raw
            other = other_raw

        refusal_rate = refusal / total if total > 0 else 0.0
        halluc_rate = halluc / total if total > 0 else 0.0

        refusal_ci_low, refusal_ci_high = wilson_ci(refusal, total)
        halluc_ci_low, halluc_ci_high = wilson_ci(halluc, total)

        rows_out.append(
            dict(
                model_name=model_name if pd.notna(model_name) else "",
                task=task if pd.notna(task) else "",
                total=total,
                refusal=refusal,
                hallucination=halluc,
                other=other,
                refusal_rate=refusal_rate,
                refusal_ci_low=refusal_ci_low,
                refusal_ci_high=refusal_ci_high,
                halluc_rate=halluc_rate,
                halluc_ci_low=halluc_ci_low,
                halluc_ci_high=halluc_ci_high,
            )
        )

    # Stable ordering: model_name then task
    rows_out.sort(key=lambda r: (str(r["model_name"]), str(r["task"])))

    # Print to stdout (same style as your current script)
    header = [
        "model_name",
        "task",
        "total",
        "refusal",
        "hallucination",
        "other",
        "refusal_rate",
        "refusal_ci_low",
        "refusal_ci_high",
        "halluc_rate",
        "halluc_ci_low",
        "halluc_ci_high",
    ]
    print(",".join(header))
    for r in rows_out:
        print(
            f'{r["model_name"]},{r["task"]},{r["total"]},{r["refusal"]},{r["hallucination"]},{r["other"]},'
            f'{r["refusal_rate"]:.3f},{r["refusal_ci_low"]:.3f},{r["refusal_ci_high"]:.3f},'
            f'{r["halluc_rate"]:.3f},{r["halluc_ci_low"]:.3f},{r["halluc_ci_high"]:.3f}'
        )

    # Write CSV
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    mode = "binary (OTHER->REFUSAL)" if args.collapse_other_to_refusal else "3-way"
    print(f"[summary] Mode: {mode}")
    print(f"[summary] Wrote summary CSV to {out_path}")


if __name__ == "__main__":
    main()
