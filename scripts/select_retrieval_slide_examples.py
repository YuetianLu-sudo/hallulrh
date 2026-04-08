#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

MODEL_PRIORITY = {
    "gemma_7b_it": 0,
    "llama3_1_8b_instruct": 1,
    "mistral_7b_instruct": 2,
    "qwen2_5_7b_instruct": 3,
}

REL_PRIORITY = {
    "company_hq": 0,
    "company_ceo": 1,
    "person_occupation": 2,
    "person_father": 3,
    "person_mother": 4,
    "person_university": 5,
    "person_plays_instrument": 6,
    "person_plays_pro_sport": 7,
    "person_plays_position_in_sport": 8,
    "landmark_in_country": 9,
    "landmark_on_continent": 10,
    "product_by_company": 11,
    "star_constellation": 12,
    "superhero_archnemesis": 13,
    "superhero_person": 14,
    "adj_antonym": 50,
    "adj_comparative": 51,
    "adj_superlative": 52,
}

def load_jsonl_gz(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_prompts(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rel = r.get("relation_key") or r.get("relation") or r.get("task")
            rid = int(r["id"])
            prompt = r.get("prompt") or r.get("question") or r.get("query") or ""
            d[(str(rel), rid)] = prompt
    return d

def short_list(xs, k=5):
    if not xs:
        return ""
    return "; ".join(xs[:k])

def sort_key(row):
    return (
        MODEL_PRIORITY.get(row["model_key"], 999),
        REL_PRIORITY.get(row["relation_key"], 999),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-run", required=True)
    ap.add_argument("--allcand-run", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)

    strict_paths = sorted(Path(args.strict_run).glob("*/ranked_lists.cosine.train.jsonl.gz"))
    all_paths = sorted(Path(args.allcand_run).glob("*/ranked_lists.cosine.all.jsonl.gz"))

    strict = {}
    for p in strict_paths:
        for r in load_jsonl_gz(p):
            key = (r["model_key"], r["relation_key"], int(r["query_id"]))
            strict[key] = r

    allcand = {}
    for p in all_paths:
        for r in load_jsonl_gz(p):
            key = (r["model_key"], r["relation_key"], int(r["query_id"]))
            allcand[key] = r

    merged = []
    for key, a in allcand.items():
        s = strict.get(key, None)
        model_key, relation_key, qid = key
        prompt = prompts.get((relation_key, qid), "")
        row = {
            "model_key": model_key,
            "relation_key": relation_key,
            "query_id": qid,
            "prompt": prompt,
            "subject": a.get("subject", s.get("subject") if s else ""),
            "gold_object": a.get("gold_object", s.get("gold_object") if s else ""),
            "strict_gold_in_candidates": (s.get("gold_in_candidates") if s else False),
            "strict_rank": (s.get("rank") if s and s.get("gold_in_candidates") else None),
            "strict_num_candidates": (s.get("num_candidates") if s else None),
            "strict_top10": short_list(s.get("top10_labels", []) if s else []),
            "all_rank": a.get("rank"),
            "all_num_candidates": a.get("num_candidates"),
            "all_top10": short_list(a.get("top10_labels", [])),
        }
        merged.append(row)

    # ---------- category 1: success ----------
    success_pool = [
        r for r in merged
        if r["strict_gold_in_candidates"] is True
        and r["strict_rank"] == 1
        and (r["strict_num_candidates"] or 0) >= 30
    ]
    success_pool = sorted(
        success_pool,
        key=lambda r: (
            sort_key(r),
            -int(r["strict_num_candidates"] or 0),
            r["all_rank"] or 9999,
        )
    )

    # ---------- category 2: failure ----------
    failure_pool = [
        r for r in merged
        if r["strict_gold_in_candidates"] is True
        and r["strict_rank"] is not None
        and (
            int(r["strict_rank"]) >= 40
            or (r["strict_num_candidates"] and int(r["strict_rank"]) / max(1, int(r["strict_num_candidates"])) >= 0.30)
        )
    ]
    failure_pool = sorted(
        failure_pool,
        key=lambda r: (
            sort_key(r),
            -(int(r["strict_rank"] or 0) / max(1, int(r["strict_num_candidates"] or 1))),
            -(int(r["strict_rank"] or 0)),
        )
    )

    # ---------- category 3: strict miss but all-cand good ----------
    gap_pool = [
        r for r in merged
        if r["strict_gold_in_candidates"] is False
        and r["all_rank"] is not None
        and int(r["all_rank"]) <= 10
    ]
    gap_pool = sorted(
        gap_pool,
        key=lambda r: (
            sort_key(r),
            int(r["all_rank"] or 9999),
        )
    )

    picked = []
    used = set()

    def pick_one(pool, category):
        for r in pool:
            k = (r["model_key"], r["relation_key"], r["query_id"])
            if k in used:
                continue
            rr = dict(r)
            rr["category"] = category
            picked.append(rr)
            used.add(k)
            return rr
        return None

    pick_one(success_pool, "success")
    pick_one(failure_pool, "failure")
    pick_one(gap_pool, "strict_miss_but_all_good")

    # save selected csv
    csv_path = outdir / "selected_examples.csv"
    fieldnames = [
        "category", "model_key", "relation_key", "query_id", "prompt", "subject", "gold_object",
        "strict_gold_in_candidates", "strict_rank", "strict_num_candidates", "strict_top10",
        "all_rank", "all_num_candidates", "all_top10",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in picked:
            w.writerow(r)

    # save broader candidate pool for manual browsing
    cand_path = outdir / "candidate_examples.csv"
    candidate_rows = []
    for category, pool in [
        ("success", success_pool[:20]),
        ("failure", failure_pool[:20]),
        ("strict_miss_but_all_good", gap_pool[:20]),
    ]:
        for r in pool:
            rr = dict(r)
            rr["category"] = category
            candidate_rows.append(rr)
    with cand_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in candidate_rows:
            w.writerow(r)

    # markdown summary
    md_path = outdir / "selected_examples.md"
    with md_path.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(f"## {r['category']}\n\n")
            f.write(f"- model: `{r['model_key']}`\n")
            f.write(f"- relation: `{r['relation_key']}`\n")
            f.write(f"- query_id: `{r['query_id']}`\n")
            f.write(f"- question: {r['prompt']}\n")
            f.write(f"- gold object: {r['gold_object']}\n")
            if r["strict_gold_in_candidates"]:
                f.write(f"- strict(train-only): rank {r['strict_rank']} / {r['strict_num_candidates']}\n")
                f.write(f"- strict top-5: {r['strict_top10']}\n")
            else:
                f.write(f"- strict(train-only): gold not in candidate bank\n")
            f.write(f"- all-candidates: rank {r['all_rank']} / {r['all_num_candidates']}\n")
            f.write(f"- all-candidates top-5: {r['all_top10']}\n\n")

    print("[done] wrote:", csv_path)
    print("[done] wrote:", cand_path)
    print("[done] wrote:", md_path)
    print("\n===== selected examples =====\n")
    print(md_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
