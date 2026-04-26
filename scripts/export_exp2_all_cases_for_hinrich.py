import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = obj
    return out


def infer_model_key(path: Path):
    return path.name.replace(".all47.judged.jsonl", "").replace(".judged.jsonl", "")


def norm_label(x):
    s = str(x).strip().upper()
    if s in {"HALL", "HALLUCINATED", "WRONG", "INCORRECT", "NONCORRECT", "NON-CORRECT"}:
        return "HALLUCINATION"
    if s in {"REFUSE", "REFUSED", "ABSTAIN", "ABSTENTION"}:
        return "REFUSAL"
    if s in {"TRUE", "RIGHT"}:
        return "CORRECT"
    return s


def one_line(x):
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\\n", " ")
    s = s.replace("\r", " ")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def choose_col(cols, must_any, avoid_any=(), prefer_any=()):
    scored = []
    for c in cols:
        cl = c.lower()
        if must_any and not any(x in cl for x in must_any):
            continue
        if avoid_any and any(x in cl for x in avoid_any):
            continue
        score = 0
        for p in prefer_any:
            if p in cl:
                score += 10
        score -= cl.count(".")
        score -= len(cl) / 200.0
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def write_outputs(df, outdir: Path, prefix: str):
    raw_path = outdir / f"{prefix}_all_cases_RAW.csv"
    browse_path = outdir / f"{prefix}_all_cases_BROWSE.csv"
    summary_path = outdir / f"{prefix}_label_summary_by_model_relation.csv"

    df.to_csv(raw_path, index=False)

    browse = df.copy()
    for c in ["question", "gold_answer", "model_answer"]:
        if c in browse.columns:
            browse[c + "_one_line"] = browse[c].map(one_line)

    preferred = [
        "model_key",
        "relation_key",
        "relation_group",
        "judge_label",
        "subject",
        "question_one_line",
        "gold_answer_one_line",
        "model_answer_one_line",
        "delta_cos",
        "answered_accuracy",
        "hall_rate_answered",
        "n_test",
    ]
    keep = [c for c in preferred if c in browse.columns]
    browse[keep].to_csv(browse_path, index=False)

    summary = (
        df.groupby(["model_key", "relation_key", "judge_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["model_key", "relation_key", "judge_label"])
    )
    summary.to_csv(summary_path, index=False)

    print(f"[write] {raw_path}  n={len(df)}")
    print(f"[write] {browse_path}  n={len(df)}")
    print(f"[write] {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat-exp", default="runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--filter-n-test-gt", type=int, default=10)
    args = ap.parse_args()

    nat_exp = Path(args.nat_exp)
    outdir = Path(args.outdir) if args.outdir else nat_exp / "experiment2_all_cases_for_hinrich"
    outdir.mkdir(parents=True, exist_ok=True)

    long_path = nat_exp / "analysis_all47" / "lre3way_long_all47.csv"
    stats_path = nat_exp / "analysis_all47" / "lre3way_behavior_plus_deltacos_all47.csv"
    judge_dir = nat_exp / "judge_all47"

    long = pd.read_csv(long_path)
    stats = pd.read_csv(stats_path)

    if "model_key" not in long.columns:
        raise KeyError(f"model_key missing in {long_path}")
    if "relation_key" not in long.columns:
        raise KeyError(f"relation_key missing in {long_path}")
    if "label" not in long.columns:
        raise KeyError(f"label missing in {long_path}")

    long = long.copy()
    long["row_id"] = long.groupby("model_key").cumcount()
    long["judge_label"] = long["label"].map(norm_label)

    if "model_key" not in stats.columns and "model_name" in stats.columns:
        stats = stats.rename(columns={"model_name": "model_key"})
    if "relation_key" not in stats.columns and "relation" in stats.columns:
        stats = stats.rename(columns={"relation": "relation_key"})

    stats["model_key"] = stats["model_key"].astype(str)
    stats["relation_key"] = stats["relation_key"].astype(str)
    stats["answered_accuracy"] = 1.0 - stats["hall_rate_answered"]

    judged_rows = []
    for p in sorted(judge_dir.glob("*.judged.jsonl")):
        mk = infer_model_key(p)
        with p.open("r", encoding="utf-8") as f:
            for row_id, line in enumerate(f):
                obj = json.loads(line)
                flat = flatten(obj)
                flat["model_key"] = mk
                flat["row_id"] = row_id
                flat["_source_file"] = str(p)
                judged_rows.append(flat)

    judged = pd.DataFrame(judged_rows)

    print("[info] long rows by model:")
    print(long.groupby("model_key").size().to_string())
    print()
    print("[info] judged rows by model:")
    print(judged.groupby("model_key").size().to_string())
    print()

    merged = long.merge(
        judged,
        on=["model_key", "row_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )

    stat_cols = ["model_key", "relation_key"]
    for c in ["relation_group", "cos_improvement", "hall_rate_answered", "answered_accuracy", "n_test"]:
        if c in stats.columns:
            stat_cols.append(c)

    merged = merged.merge(
        stats[stat_cols].drop_duplicates(),
        on=["model_key", "relation_key"],
        how="left",
        suffixes=("", "_stat"),
    )

    cols = list(merged.columns)

    question_col = choose_col(
        cols,
        must_any=["question", "prompt", "query", "input"],
        avoid_any=["judge", "reason", "confidence"],
        prefer_any=["question", "prompt"],
    )
    subject_col = choose_col(
        cols,
        must_any=["subject", "subj", "entity"],
        avoid_any=["relation", "object", "answer", "judge"],
        prefer_any=["subject"],
    )
    gold_col = choose_col(
        cols,
        must_any=["gold", "target", "correct", "object"],
        avoid_any=["model", "pred", "judge", "response", "output"],
        prefer_any=["gold_answer", "gold_object", "gold", "target"],
    )
    answer_col = choose_col(
        cols,
        must_any=["answer", "response", "completion", "output", "generation", "decoded"],
        avoid_any=["gold", "target", "correct", "judge", "label", "reason", "confidence"],
        prefer_any=["model_answer", "model_output", "response", "answer", "generation", "output"],
    )

    inferred = {
        "question_col": question_col,
        "subject_col": subject_col,
        "gold_col": gold_col,
        "answer_col": answer_col,
    }
    pd.DataFrame([inferred]).to_csv(outdir / "inferred_columns.csv", index=False)
    print("[write]", outdir / "inferred_columns.csv")
    print("[info] inferred columns:", inferred)
    print()

    out = pd.DataFrame()
    out["model_key"] = merged["model_key"]
    out["relation_key"] = merged["relation_key"]
    out["relation_group"] = merged.get("relation_group", merged.get("relation_group_stat", ""))
    out["judge_label"] = merged["judge_label"]
    out["subject"] = merged[subject_col] if subject_col else ""
    out["question"] = merged[question_col] if question_col else ""
    out["gold_answer"] = merged[gold_col] if gold_col else ""
    out["model_answer"] = merged[answer_col] if answer_col else ""
    out["delta_cos"] = merged["cos_improvement"] if "cos_improvement" in merged.columns else ""
    out["hall_rate_answered"] = merged["hall_rate_answered"] if "hall_rate_answered" in merged.columns else ""
    out["answered_accuracy"] = merged["answered_accuracy"] if "answered_accuracy" in merged.columns else ""
    out["n_test"] = merged["n_test"] if "n_test" in merged.columns else ""

    out = out.sort_values(["model_key", "relation_key", "judge_label", "subject", "question"])

    # Full all47 export.
    write_outputs(out, outdir, "all47")

    # Figure-2 subset export.
    fig2 = out[out["n_test"] > args.filter_n_test_gt].copy()
    write_outputs(fig2, outdir, "fig2_n_test_gt10")

    # Per-model BROWSE files for Figure-2 subset.
    for mk, g in fig2.groupby("model_key"):
        p = outdir / f"{mk}_fig2_n_test_gt10_all_cases_BROWSE.csv"
        g2 = g.copy()
        for c in ["question", "gold_answer", "model_answer"]:
            g2[c + "_one_line"] = g2[c].map(one_line)
        keep = [
            "model_key",
            "relation_key",
            "relation_group",
            "judge_label",
            "subject",
            "question_one_line",
            "gold_answer_one_line",
            "model_answer_one_line",
            "delta_cos",
            "answered_accuracy",
            "hall_rate_answered",
            "n_test",
        ]
        g2[keep].to_csv(p, index=False)
        print(f"[write] {p}  n={len(g2)}")

    # Label counts.
    label_counts = (
        out.groupby(["model_key", "judge_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["model_key", "judge_label"])
    )
    label_counts.to_csv(outdir / "all47_label_counts_by_model.csv", index=False)

    fig2_label_counts = (
        fig2.groupby(["model_key", "judge_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["model_key", "judge_label"])
    )
    fig2_label_counts.to_csv(outdir / "fig2_n_test_gt10_label_counts_by_model.csv", index=False)

    # Schema reference.
    with (outdir / "README.txt").open("w", encoding="utf-8") as f:
        f.write(
"""Experiment 2 complete natural-LRE package.

This folder contains all cases from the natural LRE 3-way analysis, including CORRECT, HALLUCINATION, and REFUSAL.

Recommended files:
  - fig2_n_test_gt10_all_cases_BROWSE.csv
  - fig2_n_test_gt10_label_summary_by_model_relation.csv
  - fig2_n_test_gt10_label_counts_by_model.csv
  - all47_all_cases_BROWSE.csv

BROWSE files collapse question/gold/model output to one line for spreadsheet viewing.
RAW files preserve the original model output fields.

Columns:
  model_key: model identifier
  relation_key: LRE relation
  relation_group: relation group
  judge_label: CORRECT / HALLUCINATION / REFUSAL
  subject: input subject
  question_one_line: rendered question
  gold_answer_one_line: gold object
  model_answer_one_line: model generation
  delta_cos: translation-only relation linearity score from the natural LRE analysis
  answered_accuracy: 1 - Hall/(Hall+Correct)
  hall_rate_answered: Hall/(Hall+Correct)
  n_test: held-out LRE test size for the relation
"""
        )

    print("[write]", outdir / "README.txt")
    print()
    print("=== all47 label counts ===")
    print(label_counts.to_string(index=False))
    print()
    print("=== fig2 n_test>10 label counts ===")
    print(fig2_label_counts.to_string(index=False))
    print()
    print("=== sample fig2 cases ===")
    print(fig2.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
