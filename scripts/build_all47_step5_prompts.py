import argparse
import json
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
    args = ap.parse_args()

    nat_exp = Path(args.nat_exp)
    outdir = Path(args.outdir) if args.outdir else nat_exp / "inputs_all47_step5"
    outdir.mkdir(parents=True, exist_ok=True)

    long_path = nat_exp / "analysis_all47/lre3way_long_all47.csv"
    judge_dir = nat_exp / "judge_all47"

    long = pd.read_csv(long_path).copy()

    required = ["model_key", "relation_key", "label", "domain", "relation_group"]
    missing = [c for c in required if c not in long.columns]
    if missing:
        raise KeyError(f"Missing columns in {long_path}: {missing}")

    # row_id is the alignment key between long-format labels and judged jsonl rows.
    long["row_id"] = long.groupby("model_key").cumcount()

    judged_rows = []
    for p in sorted(judge_dir.glob("*.judged.jsonl")):
        mk = infer_model_key(p)
        with p.open("r", encoding="utf-8") as f:
            for row_id, line in enumerate(f):
                x = json.loads(line)
                flat = flatten(x)

                # Avoid pandas suffixes like relation_key_x/relation_key_y.
                # The authoritative relation metadata comes from lre3way_long_all47.csv.
                for k in ["model_key", "row_id", "relation_key", "relation", "task", "relation_group", "label", "domain"]:
                    if k in flat:
                        flat[f"raw_{k}"] = flat.pop(k)

                flat["model_key"] = mk
                flat["row_id"] = row_id
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
    )

    # Sanity check: relation_key should still be unsuffixed and from the long file.
    if "relation_key" not in merged.columns:
        raise KeyError(f"relation_key missing after merge. Columns: {list(merged.columns)}")

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

    if question_col is None or subject_col is None or gold_col is None:
        print("Available columns:")
        for c in cols:
            print(c)
        raise SystemExit(
            f"Could not infer columns: question={question_col}, subject={subject_col}, gold={gold_col}"
        )

    print("[info] question_col =", question_col)
    print("[info] subject_col  =", subject_col)
    print("[info] gold_col     =", gold_col)

    for mk, g in merged.groupby("model_key"):
        path = outdir / f"{mk}.all47.for_step5.with_gold.jsonl"
        with path.open("w", encoding="utf-8") as out:
            gg = g.reset_index(drop=True)
            for i, r in gg.iterrows():
                obj = {
                    "id": int(i),
                    "example_id": str(i),
                    "model_key": str(mk),
                    "relation_key": str(r["relation_key"]),
                    "relation_name": str(r["relation_key"]),
                    "relation_group": str(r["relation_group"]),
                    "subject": str(r[subject_col]),
                    "prompt": str(r[question_col]),
                    "question": str(r[question_col]),
                    "gold_object": str(r[gold_col]),
                    "gold_answer": str(r[gold_col]),
                    "label": str(r["label"]),
                    "domain": str(r["domain"]),
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[write] {path}  rows={len(g)}")

    print()
    print("[done] all47 Step5 prompts written to:", outdir)


if __name__ == "__main__":
    main()
