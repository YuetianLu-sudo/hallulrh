import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple

LABELS = ["REFUSAL", "HALLUCINATION", "OTHER"]


def detect_column(fieldnames, candidates):
    """Return the first matching column name from candidates or None."""
    lowered = {name.lower(): name for name in fieldnames}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def summarize_group(rows: List[dict], human_col: str, gemini_col: str) -> Tuple[str, str]:
    """Compute confusion matrix + metrics for one subset of rows."""
    n = 0
    correct = 0

    # conf[human_label][gemini_label] -> count
    conf = defaultdict(lambda: defaultdict(int))

    for row in rows:
        h = (row.get(human_col, "") or "").strip().upper()
        g = (row.get(gemini_col, "") or "").strip().upper()

        if not h:
            # Skip examples without human label
            continue

        # Normalise labels
        if h not in LABELS:
            # Unknown human label -> skip
            continue
        if g not in LABELS:
            # Map anything unexpected (e.g. "UNKNOWN") to OTHER
            g = "OTHER"

        conf[h][g] += 1
        n += 1
        if h == g:
            correct += 1

    if n == 0:
        return "No examples with valid human labels.\n", "N/A"

    # Overall accuracy
    acc = correct / n

    # Confusion matrix as pretty text
    lines = []
    header = ["human \\ gemini"] + LABELS
    lines.append("\t".join(header))

    for h in LABELS:
        row_counts = [h]
        for g in LABELS:
            row_counts.append(str(conf[h][g]))
        lines.append("\t".join(row_counts))

    # Per-label metrics
    metrics_lines = []
    macro_f1_sum = 0.0
    for lbl in LABELS:
        tp = conf[lbl][lbl]
        fp = sum(conf[other][lbl] for other in LABELS if other != lbl)
        fn = sum(conf[lbl][other] for other in LABELS if other != lbl)
        denom_p = tp + fp
        denom_r = tp + fn
        prec = tp / denom_p if denom_p > 0 else 0.0
        rec = tp / denom_r if denom_r > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        macro_f1_sum += f1

        metrics_lines.append(
            f"{lbl:14s}: P={prec:5.3f}  R={rec:5.3f}  F1={f1:5.3f}  "
            f"(gold={denom_r}, pred={denom_p})"
        )

    macro_f1 = macro_f1_sum / len(LABELS)

    report_lines = []
    report_lines.append(f"# Examples with human labels: {n}")
    report_lines.append(f"# Overall accuracy: {acc:.3f}")
    report_lines.append("")
    report_lines.append("Confusion matrix (rows = human, columns = Gemini):")
    report_lines.append("\n".join(lines))
    report_lines.append("")
    report_lines.append("Per-label metrics:")
    report_lines.append("\n".join(metrics_lines))
    report_lines.append("")
    report_lines.append(
        f"Macro-F1 over {{REFUSAL, HALLUCINATION, OTHER}}: {macro_f1:.3f}"
    )
    report_lines.append("")

    short_summary = f"n={n}, acc={acc:.3f}, macro-F1={macro_f1:.3f}"
    return "\n".join(report_lines), short_summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze calibration of Gemini LM-as-judge vs. human labels."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV produced by lm_judge_gemini.py judge-csv (with human_label + gemini_label columns).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a detailed text report. If omitted, only prints to stdout.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Detect column names (robust to slightly different header names)
    # Human labels: allow several common variants.
    human_col = detect_column(
        fieldnames,
        ["human_label", "manual_label", "gold_label", "label"],
    )
    # Gemini / judge labels: allow several variants.
    gemini_col = detect_column(
        fieldnames,
        ["gemini_label", "judge_label", "llm_label"],
    )

    if human_col is None:
        raise SystemExit(
            f"Could not find a human-label column. Expected one of "
            f"['human_label', 'gold_label', 'label'] in {fieldnames}"
        )
    if gemini_col is None:
        raise SystemExit(
            f"Could not find a Gemini-label column. Expected one of "
            f"['gemini_label', 'judge_label', 'llm_label'] in {fieldnames}"
        )

    # Optional task-wise breakdown if a 'task' column exists
    task_col = detect_column(fieldnames, ["task"])
    groups = {"ALL": rows}
    if task_col is not None:
        by_task = {}
        for row in rows:
            key = (row.get(task_col, "") or "").strip() or "UNKNOWN_TASK"
            by_task.setdefault(key, []).append(row)
        groups.update(by_task)

    all_reports = []
    print("=== LM-as-judge calibration summary ===")
    print(f"Input file: {args.input}")
    print(f"Detected human label column : {human_col}")
    print(f"Detected Gemini label column: {gemini_col}")
    if task_col is not None:
        print(f"Detected task column       : {task_col}")
    print("")

    for name, subset in groups.items():
        title = f"[GROUP {name}]"
        report, short = summarize_group(subset, human_col, gemini_col)
        print(title)
        print(report)
        all_reports.append(title + "\n" + report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f_out:
            f_out.write("\n\n".join(all_reports))
        print(f"\n[analyze] Wrote detailed report to {args.output}")


if __name__ == "__main__":
    main()
