import argparse
import json
import os
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
    name = path.name
    return name.replace(".all47.judged.jsonl", "").replace(".judged.jsonl", "")


def norm_label(x):
    s = str(x).strip().upper()
    if s in {"HALL", "HALLUCINATED", "WRONG", "INCORRECT", "NONCORRECT", "NON-CORRECT"}:
        return "HALLUCINATION"
    if s in {"REFUSE", "REFUSED", "ABSTAIN", "ABSTENTION"}:
        return "REFUSAL"
    if s in {"TRUE", "RIGHT"}:
        return "CORRECT"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat-exp", default="runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--filter-n-test-gt", type=int, default=10)
    args = ap.parse_args()

    nat_exp = Path(args.nat_exp)
    outdir = Path(args.outdir) if args.outdir else nat_exp / "error_analysis_for_hinrich"
    outdir.mkdir(parents=True, exist_ok=True)

    long_path = nat_exp / "analysis_all47/lre3way_long_all47.csv"
    stats_path = nat_exp / "analysis_all47/lre3way_behavior_plus_deltacos_all47.csv"
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

    fig2_stats = stats[stats["n_test"] > args.filter_n_test_gt].copy()
    fig2_relations = set(fig2_stats["relation_key"].astype(str))

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

    merged = merged[merged["relation_key"].astype(str).isin(fig2_relations)].copy()

    label_counts = (
        merged.groupby(["model_key", "judge_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["model_key", "judge_label"])
    )
    label_counts.to_csv(outdir / "label_counts_by_model.csv", index=False)
    print("[write]", outdir / "label_counts_by_model.csv")
    print(label_counts.to_string(index=False))
    print()

    wrong = merged[merged["judge_label"] == "HALLUCINATION"].copy()
    if wrong.empty:
        raise ValueError("No HALLUCINATION rows after row-id merge; inspect label_counts_by_model.csv")

    stat_cols = ["model_key", "relation_key"]
    for c in ["relation_group", "delta_cos", "hall_rate_answered", "hall_rate_noncorrect", "n_test"]:
        if c in fig2_stats.columns:
            stat_cols.append(c)

    wrong = wrong.merge(
        fig2_stats[stat_cols].drop_duplicates(),
        on=["model_key", "relation_key"],
        how="left",
        suffixes=("", "_stat"),
    )

    # Column guessing for compact CSV.
    cols = list(wrong.columns)

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

    # Save full flattened incorrect cases regardless of column guessing.
    full_path = outdir / "all_models_incorrect_answered_cases_FULL_FLAT.csv"
    wrong.to_csv(full_path, index=False)
    print("[write]", full_path, "n=", len(wrong))

    compact = pd.DataFrame()
    compact["model_key"] = wrong["model_key"]
    compact["relation_key"] = wrong["relation_key"]
    compact["relation_group"] = wrong.get("relation_group", wrong.get("relation_group_stat", ""))
    compact["judge_label"] = wrong["judge_label"]
    compact["subject"] = wrong[subject_col] if subject_col else ""
    compact["question"] = wrong[question_col] if question_col else ""
    compact["gold_answer"] = wrong[gold_col] if gold_col else ""
    compact["model_answer"] = wrong[answer_col] if answer_col else ""
    for c in ["delta_cos", "hall_rate_answered", "n_test"]:
        compact[c] = wrong[c] if c in wrong.columns else ""

    compact = compact.sort_values(["model_key", "relation_key", "subject", "question"])
    compact_path = outdir / "all_models_incorrect_answered_cases.csv"
    compact.to_csv(compact_path, index=False)
    print("[write]", compact_path, "n=", len(compact))

    for mk, g in compact.groupby("model_key"):
        p = outdir / f"{mk}_incorrect_answered_cases.csv"
        g.to_csv(p, index=False)
        print("[write]", p, "n=", len(g))

    summary = (
        compact.groupby(["model_key", "relation_key"])
        .size()
        .reset_index(name="n_incorrect_answered")
        .sort_values(["model_key", "n_incorrect_answered", "relation_key"], ascending=[True, False, True])
    )
    summary_path = outdir / "incorrect_answered_summary_by_relation.csv"
    summary.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    top = compact.groupby(["model_key", "relation_key"], as_index=False).head(10)
    top_path = outdir / "incorrect_answered_top10_per_relation.csv"
    top.to_csv(top_path, index=False)
    print("[write]", top_path)

    # Schema reference.
    schema_path = outdir / "full_flat_columns.txt"
    with schema_path.open("w", encoding="utf-8") as f:
        for c in wrong.columns:
            f.write(c + "\n")
    print("[write]", schema_path)

    print()
    print("=== Compact sample ===")
    print(compact.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
