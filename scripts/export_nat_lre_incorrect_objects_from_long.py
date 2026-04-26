import argparse
import os
from pathlib import Path

import pandas as pd


def norm_label(x):
    s = str(x).strip().upper()
    if s in {"HALL", "HALLUCINATED", "WRONG", "INCORRECT", "NONCORRECT", "NON-CORRECT"}:
        return "HALLUCINATION"
    if s in {"REFUSE", "REFUSED", "ABSTAIN", "ABSTENTION"}:
        return "REFUSAL"
    if s in {"TRUE", "RIGHT"}:
        return "CORRECT"
    return s


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}\nAvailable columns: {list(df.columns)}")
    return None


def infer_label_col(df):
    candidates = [
        "judge_label", "label", "pred_label", "prediction", "class",
        "judge_decision", "decision", "llm_label", "llm_judge_label",
        "verdict", "annotation", "natural_label", "label_3way",
    ]
    for c in candidates:
        if c in df.columns:
            vals = df[c].dropna().map(norm_label)
            if vals.isin(["CORRECT", "HALLUCINATION", "REFUSAL"]).sum() > 0:
                return c

    best = None
    best_score = -1
    for c in df.columns:
        vals = df[c].dropna().map(norm_label)
        score = vals.isin(["CORRECT", "HALLUCINATION", "REFUSAL"]).sum()
        has_hall = (vals == "HALLUCINATION").sum()
        if score > best_score and has_hall > 0:
            best = c
            best_score = score

    if best is None:
        raise KeyError(
            "Could not infer label column. Please inspect lre3way_long_all47.csv columns/value counts."
        )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat-exp", default="runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--filter-n-test-gt", type=int, default=10)
    args = ap.parse_args()

    nat_exp = Path(args.nat_exp)
    outdir = Path(args.outdir) if args.outdir else nat_exp / "error_analysis_for_hinrich"
    outdir.mkdir(parents=True, exist_ok=True)

    long_path = nat_exp / "analysis_all47" / "lre3way_long_all47.csv"
    stats_path = nat_exp / "analysis_all47" / "lre3way_behavior_plus_deltacos_all47.csv"

    df = pd.read_csv(long_path)
    stats = pd.read_csv(stats_path)

    print(f"[info] loaded long: {long_path}  n={len(df)}")
    print(f"[info] loaded stats: {stats_path}  n={len(stats)}")

    # Normalize key columns.
    model_col = pick_col(df, ["model_key", "model_name", "model"])
    rel_col = pick_col(df, ["relation_key", "relation", "task"])
    label_col = infer_label_col(df)

    q_col = pick_col(df, ["question", "prompt", "query", "input"], required=False)
    subj_col = pick_col(df, ["subject", "subj", "entity"], required=False)
    gold_col = pick_col(df, ["gold_answer", "gold_object", "gold", "target", "object"], required=False)
    ans_col = pick_col(df, [
        "model_answer", "answer", "response", "completion", "output",
        "decoded_text", "generation", "model_output", "generated_text",
    ], required=False)

    if ans_col is None:
        raise KeyError(f"Could not find model-answer column. Available columns: {list(df.columns)}")
    if gold_col is None:
        raise KeyError(f"Could not find gold-answer column. Available columns: {list(df.columns)}")

    print("[info] inferred columns:")
    print("  model_col =", model_col)
    print("  rel_col   =", rel_col)
    print("  label_col =", label_col)
    print("  q_col     =", q_col)
    print("  subj_col  =", subj_col)
    print("  gold_col  =", gold_col)
    print("  ans_col   =", ans_col)

    df = df.copy()
    df["model_key"] = df[model_col].astype(str)
    df["relation_key"] = df[rel_col].astype(str)
    df["judge_label_norm"] = df[label_col].map(norm_label)
    df["question_out"] = df[q_col].astype(str) if q_col else ""
    df["subject_out"] = df[subj_col].astype(str) if subj_col else ""
    df["gold_out"] = df[gold_col].astype(str)
    df["model_answer_out"] = df[ans_col].astype(str)

    # Normalize stats keys.
    if "model_key" not in stats.columns and "model_name" in stats.columns:
        stats = stats.rename(columns={"model_name": "model_key"})
    if "relation_key" not in stats.columns and "relation" in stats.columns:
        stats = stats.rename(columns={"relation": "relation_key"})
    stats["model_key"] = stats["model_key"].astype(str)
    stats["relation_key"] = stats["relation_key"].astype(str)

    # Current Figure 2 relation set: n_test > 10.
    fig2_stats = stats[stats["n_test"] > args.filter_n_test_gt].copy()
    fig2_relations = set(fig2_stats["relation_key"].astype(str))
    df = df[df["relation_key"].isin(fig2_relations)].copy()

    label_counts = (
        df.groupby(["model_key", "judge_label_norm"])
        .size()
        .reset_index(name="count")
        .sort_values(["model_key", "judge_label_norm"])
    )
    label_counts.to_csv(outdir / "label_counts_by_model.csv", index=False)
    print("[write]", outdir / "label_counts_by_model.csv")
    print("\n=== label counts after n_test filter ===")
    print(label_counts.to_string(index=False))

    wrong = df[df["judge_label_norm"] == "HALLUCINATION"].copy()
    if wrong.empty:
        raise ValueError(
            "No HALLUCINATION rows found after normalization. "
            "Inspect label_counts_by_model.csv and label_col."
        )

    # Attach relation-level stats.
    keep_stats = ["model_key", "relation_key"]
    for c in ["relation_group", "delta_cos", "hall_rate_answered", "hall_rate_noncorrect", "n_test"]:
        if c in fig2_stats.columns:
            keep_stats.append(c)
    wrong = wrong.merge(
        fig2_stats[keep_stats].drop_duplicates(),
        on=["model_key", "relation_key"],
        how="left",
        suffixes=("", "_rel"),
    )

    # Construct clean output.
    out_cols = [
        "model_key",
        "relation_key",
        "relation_group",
        "subject_out",
        "question_out",
        "gold_out",
        "model_answer_out",
        "judge_label_norm",
        "delta_cos",
        "hall_rate_answered",
        "n_test",
    ]
    out_cols = [c for c in out_cols if c in wrong.columns]

    wrong_out = wrong[out_cols].copy()
    wrong_out = wrong_out.rename(columns={
        "subject_out": "subject",
        "question_out": "question",
        "gold_out": "gold_answer",
        "model_answer_out": "model_answer",
        "judge_label_norm": "judge_label",
    })
    wrong_out = wrong_out.sort_values(["model_key", "relation_key", "subject", "question"])

    combined_path = outdir / "all_models_incorrect_answered_cases.csv"
    wrong_out.to_csv(combined_path, index=False)
    print("[write]", combined_path, "n=", len(wrong_out))

    # Per-model files.
    for mk, g in wrong_out.groupby("model_key"):
        p = outdir / f"{mk}_incorrect_answered_cases.csv"
        g.to_csv(p, index=False)
        print("[write]", p, "n=", len(g))

    # Summary by relation.
    summary = (
        wrong_out.groupby(["model_key", "relation_key"])
        .size()
        .reset_index(name="n_incorrect_answered")
        .sort_values(["model_key", "n_incorrect_answered", "relation_key"], ascending=[True, False, True])
    )
    summary_path = outdir / "incorrect_answered_summary_by_relation.csv"
    summary.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    # Top examples per relation.
    top = wrong_out.groupby(["model_key", "relation_key"], as_index=False).head(10)
    top_path = outdir / "incorrect_answered_top10_per_relation.csv"
    top.to_csv(top_path, index=False)
    print("[write]", top_path)

    print("\n=== summary head ===")
    print(summary.head(80).to_string(index=False))

    print("\n=== sample wrong cases ===")
    sample_cols = [c for c in ["model_key", "relation_key", "subject", "gold_answer", "model_answer"] if c in wrong_out.columns]
    print(wrong_out[sample_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
