#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step5 (LRE Hernandez): extract hidden states and compute per-triple Δcos (gold object).

For each relation r:
  - Build vectors:
      s_i = mean pooled hidden states over subject span at subject_layer
      o_i = mean pooled hidden states over gold_object span at object_layer
  - Fit relation direction on train split:
      d_r = mean_{i in train} (o_i - s_i)
  - For each triple i:
      base_cos_i  = cos(s_i, o_i)
      cos_i       = cos(s_i + d_r, o_i)
      delta_cos_i = cos_i - base_cos_i

Output structure (per model):
  out_dir/
    per_relation/{relation_key}.csv.gz
    relation_summary.csv.gz
    per_triple.csv.gz   (concatenation of per_relation)
Resume:
  If per_relation/{relation_key}.csv.gz already exists and is non-empty, it is reused.

NOTE:
  - Requires fast tokenizer for offset_mapping.
  - If --use_chat_template is enabled, we render (system,user,assistant) via tokenizer.apply_chat_template when available.
"""

import argparse
import csv
import gzip
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
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


def load_prompts_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


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
            # If system role unsupported, merge system into user.
            msgs2 = [
                {"role": "user", "content": (system_prompt + "\n\n" + prompt).strip()},
                {"role": "assistant", "content": answer},
            ]
            return tokenizer.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=False)

    # Fallback: plain text
    return (prompt + " " + answer).strip()


def find_token_indices_for_span(
    full_text: str,
    span: str,
    offsets: List[Tuple[int, int]],
    which: str = "first",
) -> List[int]:
    if which not in ("first", "last"):
        raise ValueError(f"which must be 'first' or 'last', got {which}")

    if not span:
        raise ValueError("span is empty")

    if which == "first":
        start = full_text.find(span)
    else:
        start = full_text.rfind(span)

    if start == -1:
        raise ValueError(f"Span not found: {span!r}")

    end = start + len(span)

    idxs: List[int] = []
    for i, (s, e) in enumerate(offsets):
        # special tokens often have (0,0)
        if s == 0 and e == 0:
            continue
        if s < end and e > start:
            idxs.append(i)

    if not idxs:
        raise ValueError(f"No token offsets overlap span: {span!r}")
    return idxs


def mean_pool(hidden: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    # hidden: [seq, dim] on GPU
    return hidden[idxs, :].float().mean(dim=0).detach().cpu()


def get_subject_object_vecs_batch(
    model,
    tokenizer,
    prompts: List[str],
    subjects: List[str],
    golds: List[str],
    system_prompt: str,
    subject_layer: int,
    object_layer: int,
    device: torch.device,
    use_chat_template: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    Returns:
      subj_vecs, obj_vecs, kept_indices (indices within the batch that were successfully aligned)
    """
    assert len(prompts) == len(subjects) == len(golds)
    n = len(prompts)

    full_texts: List[str] = []
    for p, g in zip(prompts, golds):
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
        raise RuntimeError(
            "Tokenizer does not provide offset_mapping. Ensure you use a *fast* tokenizer "
            "(AutoTokenizer(..., use_fast=True))."
        )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets_batch = enc["offset_mapping"].tolist()

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = out.hidden_states
    subj_h = hidden_states[subject_layer + 1]  # [batch, seq, dim]
    obj_h  = hidden_states[object_layer + 1]   # [batch, seq, dim]

    subj_vecs: List[torch.Tensor] = []
    obj_vecs: List[torch.Tensor] = []
    kept: List[int] = []

    for i in range(n):
        full_text = full_texts[i]
        subj = normalize_text(subjects[i])
        gold = normalize_text(golds[i])
        offsets = [(int(s), int(e)) for s, e in offsets_batch[i]]

        try:
            subj_idxs = find_token_indices_for_span(full_text, subj, offsets, which="first")
            obj_idxs  = find_token_indices_for_span(full_text, gold, offsets, which="last")
        except ValueError:
            continue

        subj_vecs.append(mean_pool(subj_h[i], subj_idxs))
        obj_vecs.append(mean_pool(obj_h[i], obj_idxs))
        kept.append(i)

    return subj_vecs, obj_vecs, kept


def write_gz_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_rows_gz_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def concat_gz_csvs(out_path: Path, part_paths: List[Path], fieldnames: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for p in part_paths:
            with gzip.open(p, "rt", encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    w.writerow(row)


def summarize_existing_rel_file(
    rel_out: Path,
    n_total: int,
    model_key: str,
    model_id: str,
    rel_key: str,
    rel_group: str,
    rel_name: str,
    subject_layer: int,
    object_layer: int,
) -> Dict[str, Any]:
    rows = read_rows_gz_csv(rel_out)
    n_used = len(rows)
    if n_used == 0:
        return {
            "model_key": model_key,
            "model_id": model_id,
            "relation_key": rel_key,
            "relation_group": rel_group,
            "relation_name": rel_name,
            "n_total": n_total,
            "n_used": 0,
            "n_skipped_align": "",
            "n_train": 0,
            "n_test": 0,
            "base_cos_mean_test": float("nan"),
            "cos_mean_test": float("nan"),
            "delta_cos_mean_test": float("nan"),
            "delta_cos_median_test": float("nan"),
            "direction_norm": float("nan"),
            "subject_layer": subject_layer,
            "object_layer": object_layer,
        }

    # parse numeric
    base = []
    cosv = []
    delt = []
    split = []
    for r in rows:
        try:
            base.append(float(r["base_cos"]))
            cosv.append(float(r["cos"]))
            delt.append(float(r["delta_cos"]))
            split.append(r.get("split", ""))
        except Exception:
            continue

    is_test = [s != "train" for s in split]
    if not any(is_test):
        is_test = [True] * len(base)

    test_delta = [d for d, t in zip(delt, is_test) if t]
    test_base  = [b for b, t in zip(base, is_test) if t]
    test_cos   = [c for c, t in zip(cosv, is_test) if t]

    test_delta_sorted = sorted(test_delta)
    med = test_delta_sorted[len(test_delta_sorted)//2] if test_delta_sorted else float("nan")

    direction_norm = float(rows[0].get("direction_norm", "nan"))

    return {
        "model_key": model_key,
        "model_id": model_id,
        "relation_key": rel_key,
        "relation_group": rel_group,
        "relation_name": rel_name,
        "n_total": n_total,
        "n_used": len(base),
        "n_skipped_align": "",
        "n_train": sum(1 for s in split if s == "train"),
        "n_test": sum(1 for s in split if s != "train"),
        "base_cos_mean_test": sum(test_base)/len(test_base) if test_base else float("nan"),
        "cos_mean_test": sum(test_cos)/len(test_cos) if test_cos else float("nan"),
        "delta_cos_mean_test": sum(test_delta)/len(test_delta) if test_delta else float("nan"),
        "delta_cos_median_test": med,
        "direction_norm": direction_norm,
        "subject_layer": subject_layer,
        "object_layer": object_layer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--system_prompt", default="You are a helpful assistant. Answer with a single short phrase.")
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--subject_layer", type=int, default=None)
    ap.add_argument("--object_layer", type=int, default=None)
    args = ap.parse_args()

    prompts_path = Path(args.prompts_jsonl)
    out_dir = Path(args.out_dir)
    per_rel_dir = out_dir / "per_relation"
    per_rel_dir.mkdir(parents=True, exist_ok=True)

    examples = load_prompts_jsonl(prompts_path)

    rel2ex: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        rel2ex[str(ex.get("relation_key"))].append(ex)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[step5] Loading tokenizer/model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    if n_layers <= 0:
        raise RuntimeError("Could not read model.config.num_hidden_layers")

    subject_layer = args.subject_layer if args.subject_layer is not None else (n_layers // 2)
    object_layer = args.object_layer if args.object_layer is not None else max(subject_layer + 2, n_layers - 2)

    print(f"[step5] num_layers={n_layers} subject_layer={subject_layer} object_layer={object_layer}")

    per_triple_fields = [
        "id",
        "model_key",
        "model_id",
        "relation_key",
        "relation_group",
        "relation_name",
        "split",
        "base_cos",
        "cos",
        "delta_cos",
        "direction_norm",
        "subject_layer",
        "object_layer",
    ]
    rel_summary_fields = [
        "model_key",
        "model_id",
        "relation_key",
        "relation_group",
        "relation_name",
        "n_total",
        "n_used",
        "n_skipped_align",
        "n_train",
        "n_test",
        "base_cos_mean_test",
        "cos_mean_test",
        "delta_cos_mean_test",
        "delta_cos_median_test",
        "direction_norm",
        "subject_layer",
        "object_layer",
    ]

    rel_summ_rows: List[Dict[str, Any]] = []
    per_rel_paths: List[Path] = []

    total = sum(len(v) for v in rel2ex.values())
    pbar = tqdm(total=total, desc=f"deltacos[{args.model_key}]")

    for rel_key in sorted(rel2ex.keys()):
        exs = rel2ex[rel_key]
        n_total = len(exs)
        if n_total == 0:
            continue

        rel_group = str(exs[0].get("relation_group", ""))
        rel_name = str(exs[0].get("relation_name", ""))

        rel_out = per_rel_dir / f"{rel_key}.csv.gz"
        if rel_out.exists() and rel_out.stat().st_size > 100:
            per_rel_paths.append(rel_out)
            # add summary from existing
            rel_summ_rows.append(
                summarize_existing_rel_file(
                    rel_out=rel_out,
                    n_total=n_total,
                    model_key=args.model_key,
                    model_id=args.model_id,
                    rel_key=rel_key,
                    rel_group=rel_group,
                    rel_name=rel_name,
                    subject_layer=subject_layer,
                    object_layer=object_layer,
                )
            )
            pbar.update(n_total)
            continue

        # deterministic split per relation (matching earlier style)
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_total, generator=g).tolist()

        n_train = max(8, int(args.train_frac * n_total))
        n_train = min(n_train, max(1, n_total - 4))  # keep >=4 test when possible
        train_set = set(perm[:n_train])

        subj_vecs: List[torch.Tensor] = []
        obj_vecs: List[torch.Tensor] = []
        kept_ids: List[int] = []
        kept_is_train: List[bool] = []
        skipped_align = 0

        bs = max(1, int(args.batch_size))
        for start in range(0, n_total, bs):
            batch = exs[start:start + bs]
            prompts = [str(b.get("prompt", "")) for b in batch]
            subjects = [str(b.get("subject", "")) for b in batch]
            golds = [str(b.get("gold_object", "")) for b in batch]
            ids = [int(b.get("id")) for b in batch]

            s_vecs, o_vecs, kept_idx = get_subject_object_vecs_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                subjects=subjects,
                golds=golds,
                system_prompt=args.system_prompt,
                subject_layer=subject_layer,
                object_layer=object_layer,
                device=device,
                use_chat_template=args.use_chat_template,
            )
            skipped_align += (len(batch) - len(kept_idx))

            # keep only aligned examples, preserving mapping
            for local_i, s_vec, o_vec in zip(kept_idx, s_vecs, o_vecs):
                global_i = start + local_i
                subj_vecs.append(s_vec)
                obj_vecs.append(o_vec)
                kept_ids.append(ids[local_i])
                kept_is_train.append(global_i in train_set)

            pbar.update(len(batch))

        n_used = len(kept_ids)
        if n_used < 2:
            write_gz_csv(rel_out, per_triple_fields, [])
            per_rel_paths.append(rel_out)
            rel_summ_rows.append({
                "model_key": args.model_key,
                "model_id": args.model_id,
                "relation_key": rel_key,
                "relation_group": rel_group,
                "relation_name": rel_name,
                "n_total": n_total,
                "n_used": n_used,
                "n_skipped_align": skipped_align,
                "n_train": 0,
                "n_test": 0,
                "base_cos_mean_test": float("nan"),
                "cos_mean_test": float("nan"),
                "delta_cos_mean_test": float("nan"),
                "delta_cos_median_test": float("nan"),
                "direction_norm": float("nan"),
                "subject_layer": subject_layer,
                "object_layer": object_layer,
            })
            continue

        subj = torch.stack(subj_vecs, dim=0).float()
        obj = torch.stack(obj_vecs, dim=0).float()

        is_train = torch.tensor(kept_is_train, dtype=torch.bool)
        if int(is_train.sum().item()) < 1:
            is_train[:] = True

        direction = (obj[is_train] - subj[is_train]).mean(dim=0)
        direction_norm = float(direction.norm().item())

        base_cos = F.cosine_similarity(subj, obj, dim=-1)
        cosv = F.cosine_similarity(subj + direction, obj, dim=-1)
        delta = cosv - base_cos

        rows: List[Dict[str, Any]] = []
        for i in range(n_used):
            rows.append({
                "id": kept_ids[i],
                "model_key": args.model_key,
                "model_id": args.model_id,
                "relation_key": rel_key,
                "relation_group": rel_group,
                "relation_name": rel_name,
                "split": "train" if bool(is_train[i].item()) else "test",
                "base_cos": float(base_cos[i].item()),
                "cos": float(cosv[i].item()),
                "delta_cos": float(delta[i].item()),
                "direction_norm": direction_norm,
                "subject_layer": subject_layer,
                "object_layer": object_layer,
            })

        write_gz_csv(rel_out, per_triple_fields, rows)
        per_rel_paths.append(rel_out)

        is_test = ~is_train
        if int(is_test.sum().item()) == 0:
            is_test = is_train

        test_base = base_cos[is_test]
        test_cos = cosv[is_test]
        test_delta = delta[is_test]

        rel_summ_rows.append({
            "model_key": args.model_key,
            "model_id": args.model_id,
            "relation_key": rel_key,
            "relation_group": rel_group,
            "relation_name": rel_name,
            "n_total": n_total,
            "n_used": n_used,
            "n_skipped_align": skipped_align,
            "n_train": int(is_train.sum().item()),
            "n_test": int(is_test.sum().item()),
            "base_cos_mean_test": float(test_base.mean().item()),
            "cos_mean_test": float(test_cos.mean().item()),
            "delta_cos_mean_test": float(test_delta.mean().item()),
            "delta_cos_median_test": float(test_delta.median().item()),
            "direction_norm": direction_norm,
            "subject_layer": subject_layer,
            "object_layer": object_layer,
        })

    pbar.close()

    rel_sum_path = out_dir / "relation_summary.csv.gz"
    write_gz_csv(rel_sum_path, rel_summary_fields, rel_summ_rows)
    print(f"[step5] wrote: {rel_sum_path}")

    per_triple_path = out_dir / "per_triple.csv.gz"
    concat_gz_csvs(per_triple_path, per_rel_paths, per_triple_fields)
    print(f"[step5] wrote: {per_triple_path}")
    print("[step5] done.")


if __name__ == "__main__":
    main()
