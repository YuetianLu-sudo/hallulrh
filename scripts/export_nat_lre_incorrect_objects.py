import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import pandas as pd


def norm_ws(x: str) -> str:
    return " ".join(str(x).strip().split())


def norm_key(x: str) -> str:
    return str(x).strip().lower()


def pick_first(obj, keys):
    for k in keys:
        if isinstance(obj, dict) and k in obj and obj[k] is not None:
            return obj[k]
    return None


def maybe_parse_json(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        y = json.loads(s)
        if isinstance(y, dict):
            return y
    except Exception:
        return None
    return None


def get_nested_label(obj):
    # Direct common fields
    label = pick_first(obj, [
        "judge_label", "label", "judgement", "judgment",
        "gemini_label", "llm_judge_label", "pred_label",
    ])
    if label is not None:
        return str(label).strip().upper()

    # Nested dict fields
    for k in ["judge", "judge_output", "judge_json", "gemini", "eval", "annotation"]:
        y = maybe_parse_json(obj.get(k)) if isinstance(obj, dict) else None
        if y:
            label = pick_first(y, ["label", "judge_label", "prediction"])
            if label is not None:
                return str(label).strip().upper()

    # Sometimes raw text contains JSON
    for k in ["judge_raw", "raw_judge", "judge_response", "response_json"]:
        y = maybe_parse_json(obj.get(k)) if isinstance(obj, dict) else None
        if y:
            label = pick_first(y, ["label", "judge_label", "prediction"])
            if label is not None:
                return str(label).strip().upper()

    return None


def get_model_answer(obj):
    ans = pick_first(obj, [
        "model_answer", "answer", "response", "completion",
        "output", "decoded_text", "generation", "model_output",
        "generated_text", "prediction",
    ])
    if ans is not None:
        return str(ans)

    # Avoid returning judge output as answer
    return ""


def get_question(obj):
    q = pick_first(obj, ["question", "prompt", "input", "query"])
    return "" if q is None else str(q)


def get_gold(obj):
    g = pick_first(obj, ["gold_answer", "gold_object", "gold", "target", "object"])
    return "" if g is None else str(g)


def get_relation(obj):
    r = pick_first(obj, ["relation_key", "relation", "task", "relation_name"])
    return "" if r is None else str(r)


def get_subject(obj):
    s = pick_first(obj, ["subject", "subj", "entity"])
    return "" if s is None else str(s)


def infer_model_key_from_path(path: str) -> str:
    name = Path(path).name
    name = name.replace(".all47.judged.jsonl", "")
    name = name.replace(".judged.jsonl", "")
    name = name.replace(".with_judge.csv", "")
    return name


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            x["_source_file"] = str(path)
            x["_line_no"] = line_no
            rows.append(x)
    return rows


def load_judged_file(path):
    path = str(path)
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        rows = df.to_dict("records")
        for i, x in enumerate(rows, start=1):
            x["_source_file"] = path
            x["_line_no"] = i
        return rows
    raise ValueError(f"Unsupported judged file: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat-exp", default="runs/experiments/fig2_lre21_nat_3wayjudge_20260126_161209")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--filter-n-test-gt", type=int, default=10)
    args = ap.parse_args()

    nat_exp = Path(args.nat_exp)
    outdir = Path(args.outdir) if args.outdir else nat_exp / "error_analysis_for_hinrich"
    outdir.mkdir(parents=True, exist_ok=True)

    # Prefer all47 because current Figure 2 is based on analysis_all47 filtered to n_test > 10.
    judged_files = sorted(glob(str(nat_exp / "judge_all47" / "*.judged.jsonl")))
    if not judged_files:
        judged_files = sorted(glob(str(nat_exp / "judge" / "*.judged.jsonl")))
    if not judged_files:
        judged_files = sorted(glob(str(nat_exp / "judge" / "*.with_judge.csv")))
    if not judged_files:
        raise FileNotFoundError(f"No judged files found under {nat_exp}/judge_all47 or {nat_exp}/judge")

    print("[info] judged files:")
    for p in judged_files:
        print("  ", p)

    # Load relation-level stats and define current Figure-2 relation set.
    stats_path = nat_exp / "analysis_all47" / "lre3way_behavior_plus_deltacos_all47.csv"
    stats = pd.read_csv(stats_path)
    if "model_key" not in stats.columns and "model_name" in stats.columns:
        stats = stats.rename(columns={"model_name": "model_key"})
    if "relation" in stats.columns and "relation_key" not in stats.columns:
        stats = stats.rename(columns={"relation": "relation_key"})

    fig2_stats = stats[stats["n_test"] > args.filter_n_test_gt].copy()
    fig2_relations = set(fig2_stats["relation_key"].astype(str))

    # Load input metadata with gold/subject/question if needed.
    input_rows = []
    for p in sorted((nat_exp / "inputs").glob("*.with_gold.jsonl")):
        mk = p.name.split(".for_3way_judge")[0]
        for x in load_jsonl(p):
            input_rows.append({
                "model_key": mk,
                "relation_key": get_relation(x),
                "question_key": norm_ws(get_question(x)),
                "question_meta": get_question(x),
                "subject_meta": get_subject(x),
                "gold_meta": get_gold(x),
                "relation_group_meta": x.get("relation_group", ""),
                "template_meta": x.get("template", ""),
            })
    inputs = pd.DataFrame(input_rows).drop_duplicates(
        subset=["model_key", "relation_key", "question_key"]
    )

    all_wrong = []
    label_counter = {}

    for jf in judged_files:
        mk_from_file = infer_model_key_from_path(jf)
        rows = load_judged_file(jf)

        normalized = []
        for x in rows:
            mk = str(pick_first(x, ["model_key", "model_name", "model"]) or mk_from_file)
            rk = get_relation(x)
            q = get_question(x)
            label = get_nested_label(x)
            ans = get_model_answer(x)
            gold = get_gold(x)
            subj = get_subject(x)

            normalized.append({
                "model_key": mk,
                "relation_key": rk,
                "question": q,
                "question_key": norm_ws(q),
                "subject": subj,
                "gold_answer": gold,
                "model_answer": ans,
                "judge_label": label,
                "source_file": x.get("_source_file"),
                "line_no": x.get("_line_no"),
            })

        df = pd.DataFrame(normalized)
        df["judge_label"] = df["judge_label"].astype(str).str.upper()
        label_counter[mk_from_file] = df["judge_label"].value_counts(dropna=False).to_dict()

        # Attach missing metadata from inputs.
        df = df.merge(
            inputs,
            on=["model_key", "relation_key", "question_key"],
            how="left",
            validate="many_to_one",
        )
        df["question"] = df["question"].where(df["question"].astype(str).str.len() > 0, df["question_meta"])
        df["subject"] = df["subject"].where(df["subject"].astype(str).str.len() > 0, df["subject_meta"])
        df["gold_answer"] = df["gold_answer"].where(df["gold_answer"].astype(str).str.len() > 0, df["gold_meta"])

        # Restrict to the relation set used in current Figure 2.
        df = df[df["relation_key"].astype(str).isin(fig2_relations)].copy()

        wrong = df[df["judge_label"].isin(["HALLUCINATION"])].copy()

        if wrong.empty:
            print(f"[warn] no HALLUCINATION rows found for {mk_from_file}. Label counts: {label_counter[mk_from_file]}")
            continue

        # Attach relation-level delta_cos and rates.
        st = fig2_stats.copy()
        wrong = wrong.merge(
            st,
            on=["model_key", "relation_key"],
            how="left",
            suffixes=("", "_rel"),
        )

        keep = [
            "model_key",
            "relation_key",
            "relation_group",
            "subject",
            "question",
            "gold_answer",
            "model_answer",
            "judge_label",
            "delta_cos",
            "hall_rate_answered",
            "n_test",
            "source_file",
            "line_no",
        ]
        keep = [c for c in keep if c in wrong.columns]
        wrong = wrong[keep].sort_values(["relation_key", "subject", "question"])

        out_csv = outdir / f"{mk_from_file}_incorrect_answered_cases.csv"
        wrong.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}  n={len(wrong)}")
        all_wrong.append(wrong)

    label_df = pd.DataFrame([
        {"model_file": mk, "label": lab, "count": cnt}
        for mk, d in label_counter.items()
        for lab, cnt in d.items()
    ])
    label_df.to_csv(outdir / "label_counts_by_file.csv", index=False)
    print(f"[write] {outdir / 'label_counts_by_file.csv'}")

    if not all_wrong:
        raise ValueError(
            "No incorrect answered cases were exported. "
            "Inspect label_counts_by_file.csv and judged-file schema."
        )

    combined = pd.concat(all_wrong, ignore_index=True)
    combined.to_csv(outdir / "all_models_incorrect_answered_cases.csv", index=False)
    print(f"[write] {outdir / 'all_models_incorrect_answered_cases.csv'}  n={len(combined)}")

    summary = (
        combined.groupby(["model_key", "relation_key"])
        .size()
        .reset_index(name="n_incorrect_answered")
        .sort_values(["model_key", "n_incorrect_answered", "relation_key"], ascending=[True, False, True])
    )
    summary.to_csv(outdir / "incorrect_answered_summary_by_relation.csv", index=False)
    print(f"[write] {outdir / 'incorrect_answered_summary_by_relation.csv'}")

    top = combined.groupby(["model_key", "relation_key"], as_index=False).head(10)
    top.to_csv(outdir / "incorrect_answered_top10_per_relation.csv", index=False)
    print(f"[write] {outdir / 'incorrect_answered_top10_per_relation.csv'}")

    print()
    print("=== Label counts ===")
    print(label_df.to_string(index=False))
    print()
    print("=== Summary head ===")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
