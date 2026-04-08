#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import gzip
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
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


def choose_first(rec: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = rec.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return default


def load_relation_set(path: Path) -> List[str]:
    rels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rels.append(line)
    return rels


def load_prompts_jsonl(path: Path, relset: set) -> Dict[str, List[Dict[str, Any]]]:
    rel2ex = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rel = choose_first(r, ["relation_key", "relation", "task"], "")
            if rel not in relset:
                continue
            ex = {
                "id": int(r["id"]),
                "relation_key": rel,
                "prompt": choose_first(r, ["prompt", "question", "query"]),
                "subject": choose_first(r, ["subject"]),
                "gold_object": choose_first(r, ["gold_object", "answer", "object"]),
            }
            rel2ex[rel].append(ex)
    for rel in rel2ex:
        rel2ex[rel] = sorted(rel2ex[rel], key=lambda x: x["id"])
    return rel2ex


def read_gz_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def load_split_info(base_step5_dir: Path, relset: List[str]) -> Tuple[Dict[Tuple[str, int], str], str, int, int, str]:
    split_map: Dict[Tuple[str, int], str] = {}
    model_id = None
    subject_layer = None
    object_layer = None
    model_key = None

    for rel in relset:
        p = base_step5_dir / "per_relation" / f"{rel}.csv.gz"
        if not p.exists():
            raise FileNotFoundError(f"missing per_relation file: {p}")
        rows = read_gz_csv(p)
        if not rows:
            continue
        for row in rows:
            rid = int(row["id"])
            split_map[(rel, rid)] = row["split"]
        if model_id is None:
            model_id = rows[0].get("model_id", None)
            model_key = rows[0].get("model_key", None)
            subject_layer = int(float(rows[0]["subject_layer"]))
            object_layer = int(float(rows[0]["object_layer"]))

    if model_id is None or subject_layer is None or object_layer is None:
        raise RuntimeError("could not infer model_id/layers from base step5 dir")

    return split_map, model_id, subject_layer, object_layer, model_key


def render_with_chat_template(tokenizer, prompt: str, answer: str, system_prompt: str) -> str:
    prompt = normalize_text(prompt).strip()
    answer = normalize_text(answer).strip()
    system_prompt = normalize_text(system_prompt).strip()

    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            msgs2 = [
                {"role": "user", "content": (system_prompt + "\n\n" + prompt).strip()},
                {"role": "assistant", "content": answer},
            ]
            return tokenizer.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=False)

    return (prompt + " " + answer).strip()


def find_token_indices_for_span(full_text: str, span: str, offsets: List[Tuple[int, int]], which: str = "first") -> List[int]:
    if which not in ("first", "last"):
        raise ValueError(which)
    span = normalize_text(span)
    start = full_text.find(span) if which == "first" else full_text.rfind(span)
    if start == -1:
        raise ValueError(f"span not found: {span!r}")
    end = start + len(span)
    idxs = []
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if s < end and e > start:
            idxs.append(i)
    if not idxs:
        raise ValueError(f"no overlap for span: {span!r}")
    return idxs


def mean_pool(hidden: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    return hidden[idxs, :].float().mean(dim=0).detach().cpu()


def get_subject_object_vecs_batch(
    model,
    tokenizer,
    batch_exs: List[Dict[str, Any]],
    system_prompt: str,
    subject_layer: int,
    object_layer: int,
    device: torch.device,
    use_chat_template: bool,
):
    full_texts = []
    for ex in batch_exs:
        p = ex["prompt"]
        g = ex["gold_object"]
        if use_chat_template:
            full_texts.append(render_with_chat_template(tokenizer, p, g, system_prompt))
        else:
            full_texts.append((normalize_text(p).rstrip() + " " + normalize_text(g).strip()).strip())

    enc = tokenizer(
        full_texts,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
    )
    if "offset_mapping" not in enc:
        raise RuntimeError("tokenizer does not provide offset_mapping")

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets_batch = enc["offset_mapping"].tolist()

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    subj_h = out.hidden_states[subject_layer + 1]
    obj_h = out.hidden_states[object_layer + 1]

    kept = []
    for i, ex in enumerate(batch_exs):
        full_text = full_texts[i]
        subj = normalize_text(ex["subject"])
        gold = normalize_text(ex["gold_object"])
        offsets = [(int(s), int(e)) for s, e in offsets_batch[i]]
        try:
            subj_idxs = find_token_indices_for_span(full_text, subj, offsets, which="first")
            obj_idxs = find_token_indices_for_span(full_text, gold, offsets, which="last")
        except Exception:
            continue
        kept.append({
            **ex,
            "s_vec": mean_pool(subj_h[i], subj_idxs),
            "o_vec": mean_pool(obj_h[i], obj_idxs),
        })
    return kept


def build_object_prototypes(exs: List[Dict[str, Any]]) -> Tuple[List[str], torch.Tensor]:
    obj2vecs = defaultdict(list)
    for ex in exs:
        obj2vecs[ex["gold_object"]].append(ex["o_vec"])
    labels = sorted(obj2vecs.keys())
    mats = []
    for lab in labels:
        mats.append(torch.stack(obj2vecs[lab], dim=0).mean(dim=0))
    mat = torch.stack(mats, dim=0).float()
    return labels, mat


def fit_affine_ridge(train_exs: List[Dict[str, Any]], ridge_lambda: float):
    X = torch.stack([x["s_vec"] for x in train_exs], dim=0).float()  # [n, d]
    Y = torch.stack([x["o_vec"] for x in train_exs], dim=0).float()  # [n, d]

    x_mean = X.mean(dim=0)
    y_mean = Y.mean(dim=0)
    Xc = X - x_mean
    Yc = Y - y_mean

    # Dual ridge solution:
    # B = Xc^T (Xc Xc^T + lambda I)^-1 Yc
    K = Xc @ Xc.T
    n = K.shape[0]
    K = K + ridge_lambda * torch.eye(n, dtype=K.dtype)

    A = torch.linalg.solve(K, Yc)   # [n, d]
    return {
        "Xc": Xc,           # [n, d]
        "A": A,             # [n, d]
        "x_mean": x_mean,   # [d]
        "y_mean": y_mean,   # [d]
    }


def apply_affine_ridge(state, x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    xc = x - state["x_mean"]              # [d]
    k = state["Xc"] @ xc                  # [n]
    yhat = k @ state["A"] + state["y_mean"]   # [d]
    return yhat


def rank_candidates(inferred: torch.Tensor, cand_labels: List[str], cand_mat: torch.Tensor, metric: str):
    inferred = inferred.float()
    if metric == "cosine":
        scores = F.cosine_similarity(cand_mat, inferred.unsqueeze(0), dim=1)
        order = torch.argsort(scores, descending=True)
        ordered_scores = scores[order].tolist()
    elif metric == "euclidean":
        dists = torch.norm(cand_mat - inferred.unsqueeze(0), dim=1)
        scores = -dists
        order = torch.argsort(scores, descending=True)
        ordered_scores = scores[order].tolist()
    else:
        raise ValueError(metric)

    ordered_labels = [cand_labels[i] for i in order.tolist()]
    return ordered_labels, ordered_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--base-step5-dir", required=True)
    ap.add_argument("--relation-set", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--candidate-source", choices=["train", "all"], default="train")
    ap.add_argument("--query-mode", choices=["translated", "subject_only", "affine_full"], default="translated")
    ap.add_argument("--ridge-lambda", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--system-prompt", default="You are a helpful assistant. Answer with a single short phrase.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    relset = load_relation_set(Path(args.relation_set))
    relset_set = set(relset)

    split_map, model_id, subject_layer, object_layer, model_key = load_split_info(Path(args.base_step5_dir), relset)

    print(f"[load] model_key={model_key}")
    print(f"[load] model_id={model_id}")
    print(f"[load] subject_layer={subject_layer} object_layer={object_layer}")
    print(f"[load] query_mode={args.query_mode} metric={args.metric} candidate_source={args.candidate_source}")

    rel2ex = load_prompts_jsonl(Path(args.prompts), relset_set)
    total_prompts = sum(len(v) for v in rel2ex.values())
    print(f"[load] prompts read for relset = {total_prompts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    model.eval()

    per_relation_rows = []
    ranked_path = outdir / f"ranked_lists.{args.metric}.{args.candidate_source}.{args.query_mode}.jsonl.gz"
    ranked_f = gzip.open(ranked_path, "wt", encoding="utf-8")

    overall_rr = []
    overall_h1 = []
    overall_h10 = []
    overall_n_total = 0
    overall_n_covered = 0

    try:
        for rel in relset:
            exs = rel2ex.get(rel, [])
            exs = [ex for ex in exs if (rel, ex["id"]) in split_map]
            if not exs:
                continue

            enriched = []
            bs = max(1, args.batch_size)
            for start in range(0, len(exs), bs):
                batch = exs[start:start+bs]
                kept = get_subject_object_vecs_batch(
                    model=model,
                    tokenizer=tokenizer,
                    batch_exs=batch,
                    system_prompt=args.system_prompt,
                    subject_layer=subject_layer,
                    object_layer=object_layer,
                    device=device,
                    use_chat_template=args.use_chat_template,
                )
                for x in kept:
                    x["split"] = split_map[(rel, x["id"])]
                enriched.extend(kept)

            if not enriched:
                continue

            train_exs = [x for x in enriched if x["split"] == "train"]
            test_exs = [x for x in enriched if x["split"] != "train"]

            if not train_exs or not test_exs:
                continue

            subj_train = torch.stack([x["s_vec"] for x in train_exs], dim=0).float()
            obj_train = torch.stack([x["o_vec"] for x in train_exs], dim=0).float()
            direction = (obj_train - subj_train).mean(dim=0)
            affine_state = fit_affine_ridge(train_exs, args.ridge_lambda) if args.query_mode == "affine_full" else None

            source_exs = train_exs if args.candidate_source == "train" else enriched
            cand_labels, cand_mat = build_object_prototypes(source_exs)
            cand_set = set(cand_labels)

            rr = []
            h1 = []
            h10 = []
            ranks = []

            for ex in test_exs:
                overall_n_total += 1
                if args.query_mode == "subject_only":
                    inferred = ex["s_vec"].float()
                elif args.query_mode == "translated":
                    inferred = ex["s_vec"].float() + direction
                elif args.query_mode == "affine_full":
                    inferred = apply_affine_ridge(affine_state, ex["s_vec"])
                else:
                    raise ValueError(args.query_mode)

                gold = ex["gold_object"]

                if gold not in cand_set:
                    rec = {
                        "model_key": model_key,
                        "relation_key": rel,
                        "query_id": ex["id"],
                        "subject": ex["subject"],
                        "gold_object": gold,
                        "metric": args.metric,
                        "candidate_source": args.candidate_source,
                        "query_mode": args.query_mode,
                        "gold_in_candidates": False,
                        "num_candidates": len(cand_labels),
                    }
                    ranked_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                overall_n_covered += 1
                ordered_labels, ordered_scores = rank_candidates(inferred, cand_labels, cand_mat, args.metric)
                rank = ordered_labels.index(gold) + 1

                rr_i = 1.0 / rank
                h1_i = 1.0 if rank <= 1 else 0.0
                h10_i = 1.0 if rank <= 10 else 0.0

                rr.append(rr_i)
                h1.append(h1_i)
                h10.append(h10_i)
                ranks.append(rank)

                overall_rr.append(rr_i)
                overall_h1.append(h1_i)
                overall_h10.append(h10_i)

                rec = {
                    "model_key": model_key,
                    "relation_key": rel,
                    "query_id": ex["id"],
                    "subject": ex["subject"],
                    "gold_object": gold,
                    "metric": args.metric,
                    "candidate_source": args.candidate_source,
                    "query_mode": args.query_mode,
                    "gold_in_candidates": True,
                    "num_candidates": len(cand_labels),
                    "rank": rank,
                    "top10_labels": ordered_labels[:10],
                    "top10_scores": [float(x) for x in ordered_scores[:10]],
                    "ranking_labels": ordered_labels,
                    "ranking_scores": [float(x) for x in ordered_scores],
                }
                ranked_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            row = {
                "model_key": model_key,
                "relation_key": rel,
                "metric": args.metric,
                "candidate_source": args.candidate_source,
                "query_mode": args.query_mode,
                "n_train": len(train_exs),
                "n_test_total": len(test_exs),
                "n_test_covered": len(rr),
                "coverage": (len(rr) / len(test_exs)) if test_exs else float("nan"),
                "num_candidates": len(cand_labels),
                "hits@1": (sum(h1) / len(h1)) if h1 else float("nan"),
                "hits@10": (sum(h10) / len(h10)) if h10 else float("nan"),
                "mrr": (sum(rr) / len(rr)) if rr else float("nan"),
                "mean_rank": (sum(ranks) / len(ranks)) if ranks else float("nan"),
                "median_rank": (float(sorted(ranks)[len(ranks)//2]) if ranks else float("nan")),
                "direction_norm": float(direction.norm().item()),
            }
            per_relation_rows.append(row)
            print(
                f"[rel] {rel}: n_train={row['n_train']} n_test_total={row['n_test_total']} "
                f"covered={row['n_test_covered']} hits@1={row['hits@1']} hits@10={row['hits@10']} mrr={row['mrr']}"
            )

    finally:
        ranked_f.close()

    csv_path = outdir / f"retrieval_by_relation.{args.metric}.{args.candidate_source}.{args.query_mode}.csv"
    if per_relation_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_relation_rows[0].keys()))
            writer.writeheader()
            for row in per_relation_rows:
                writer.writerow(row)

    overall = {
        "model_key": model_key,
        "metric": args.metric,
        "candidate_source": args.candidate_source,
        "query_mode": args.query_mode,
        "n_test_total": overall_n_total,
        "n_test_covered": overall_n_covered,
        "coverage": (overall_n_covered / overall_n_total) if overall_n_total else float("nan"),
        "hits@1": (sum(overall_h1) / len(overall_h1)) if overall_h1 else float("nan"),
        "hits@10": (sum(overall_h10) / len(overall_h10)) if overall_h10 else float("nan"),
        "mrr": (sum(overall_rr) / len(overall_rr)) if overall_rr else float("nan"),
        "ranked_lists_path": str(ranked_path),
        "per_relation_csv": str(csv_path),
    }

    overall_path = outdir / f"retrieval_overall.{args.metric}.{args.candidate_source}.{args.query_mode}.json"
    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("[done] wrote:", csv_path)
    print("[done] wrote:", ranked_path)
    print("[done] wrote:", overall_path)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
