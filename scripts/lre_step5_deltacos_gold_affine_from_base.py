#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-relation affine cosine-improvement on LRE (gold objects),
while *reusing the exact aligned examples + train/test split* from an existing
translation-only Step5 output.

We fit:    o_hat = W s + b   (ridge regression, dual form; trained on TRAIN)
Metric:    Δcos_aff = cos(o_hat, o) - cos(s, o)    (evaluated on TEST)

Key design:
- We read ids+split (+stored base_cos) from BASE_STEP5_DIR/per_relation/<rel>.csv.gz,
  so we match the same aligned set as your existing Δcos run.
- We auto-detect whether to use tokenizer.apply_chat_template by checking which
  mode best reproduces stored base_cos on a small calibration subset.

Outputs:
  outdir/relation_summary_affine.csv.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTION_KEYS = ["question", "prompt", "input", "query", "text"]
GOLD_KEYS = ["gold_object", "gold_answer", "object", "answer", "target", "label"]


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


def open_text(path: str, mode: str = "rt", encoding: str = "utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return open(path, mode, encoding=encoding)


def _choose_first_str(rec: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _norm_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        # avoid "12.0"
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v).strip() or None


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


def find_token_indices_for_span(
    full_text: str,
    span: str,
    offsets: List[Tuple[int, int]],
    which: str = "first",
) -> List[int]:
    if which not in ("first", "last"):
        raise ValueError("which must be 'first' or 'last'")
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
        if s == 0 and e == 0:
            continue
        if s < end and e > start:
            idxs.append(i)

    if not idxs:
        raise ValueError(f"No token offsets overlap span: {span!r}")
    return idxs


def mean_pool(hidden: torch.Tensor, idxs: List[int]) -> torch.Tensor:
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
        raise RuntimeError("Need fast tokenizer with offset_mapping (use_fast=True).")

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets_batch = enc["offset_mapping"].tolist()

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hs = out.hidden_states
    subj_h = hs[subject_layer + 1]  # [B,S,D]
    obj_h  = hs[object_layer + 1]   # [B,S,D]

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
        except Exception:
            continue

        subj_vecs.append(mean_pool(subj_h[i], subj_idxs))
        obj_vecs.append(mean_pool(obj_h[i], obj_idxs))
        kept.append(i)

    return subj_vecs, obj_vecs, kept


def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = np.sum(A * B, axis=1)
    den = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + 1e-12
    return num / den


def ridge_affine_predict_dual(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float) -> np.ndarray:
    # Fit affine: y = x W^T + b with ridge via dual form.
    mu_x = Xtr.mean(axis=0)
    mu_y = Ytr.mean(axis=0)

    Xc = Xtr - mu_x
    Yc = Ytr - mu_y

    n = Xc.shape[0]
    G = Xc @ Xc.T
    if lam > 0:
        G = G + lam * np.eye(n, dtype=np.float32)

    # Alpha = (G)^-1 Yc   => shape [n, d]
    Alpha = np.linalg.solve(G, Yc)

    K = (Xte - mu_x) @ Xc.T   # [n_test, n_train]
    Ypred = K @ Alpha + mu_y  # [n_test, d]
    return Ypred


def load_relset(path: str) -> List[str]:
    rels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                rels.append(s)
    return rels


def detect_chat_template_mode(
    model,
    tokenizer,
    id2rec: Dict[str, Dict[str, Any]],
    base_step5_dir: str,
    rels: List[str],
    system_prompt: str,
    subject_layer: int,
    object_layer: int,
    device: torch.device,
    batch_size: int,
    max_calib: int = 16,
) -> bool:
    # Pick one relation with enough examples for calibration
    calib_rel = None
    base_df = None
    for r in rels:
        p = os.path.join(base_step5_dir, "per_relation", f"{r}.csv.gz")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, compression="gzip")
        if len(df) >= 8:
            calib_rel = r
            base_df = df
            break
    if calib_rel is None or base_df is None:
        print("[warn] could not find a suitable relation for chat-template detection; defaulting to OFF", file=sys.stderr)
        return False

    df = base_df.head(max_calib).copy()
    ids = [str(_norm_id(x)) for x in df["id"].tolist()]
    stored_base = df["base_cos"].astype(float).to_numpy()

    prompts, subjects, golds = [], [], []
    keep_stored = []
    for i, ex_id in enumerate(ids):
        rec = id2rec.get(ex_id)
        if rec is None:
            continue
        q = _choose_first_str(rec, QUESTION_KEYS) or ""
        subj = str(rec.get("subject") or "")
        gold = _choose_first_str(rec, GOLD_KEYS) or ""
        if not q or not subj or not gold:
            continue
        prompts.append(q)
        subjects.append(subj)
        golds.append(gold)
        keep_stored.append(stored_base[i])

    if len(prompts) < 4:
        print("[warn] too few calib examples; defaulting to OFF", file=sys.stderr)
        return False

    def eval_mode(use_chat: bool) -> float:
        # Run once (small batch)
        s_vecs, o_vecs, kept = get_subject_object_vecs_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            subjects=subjects,
            golds=golds,
            system_prompt=system_prompt,
            subject_layer=subject_layer,
            object_layer=object_layer,
            device=device,
            use_chat_template=use_chat,
        )
        if not kept:
            return float("inf")
        S = np.stack([s_vecs[i].numpy() for i in range(len(s_vecs))], axis=0).astype(np.float32)
        O = np.stack([o_vecs[i].numpy() for i in range(len(o_vecs))], axis=0).astype(np.float32)
        pred_base = cosine_rows(S, O)
        # align stored list to kept (here kept are local indices into prompts list)
        stored = np.array([keep_stored[j] for j in kept], dtype=np.float32)
        return float(np.mean(np.abs(pred_base - stored)))

    mae_off = eval_mode(False)
    mae_on  = eval_mode(True)

    use_chat = mae_on < mae_off
    print(f"[detect] calib_rel={calib_rel} base_cos MAE: chat=ON {mae_on:.6g} vs OFF {mae_off:.6g} -> choose {'ON' if use_chat else 'OFF'}", file=sys.stderr)
    return use_chat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="LRE prompts JSONL (q-only) with gold_object.")
    ap.add_argument("--base-step5-dir", required=True, help="Existing translation-only Step5 dir for this model_key.")
    ap.add_argument("--relation-set", required=True, help="Relation set file (one relation_key per line).")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ridge-lambda", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--system-prompt", default="You are a helpful assistant. Answer with a single short phrase.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # read base relation_summary for model_id + layers
    base_sum_path = os.path.join(args.base_step5_dir, "relation_summary.csv.gz")
    if not os.path.exists(base_sum_path):
        raise SystemExit(f"[error] missing base relation_summary.csv.gz: {base_sum_path}")
    base_sum = pd.read_csv(base_sum_path, compression="gzip")
    if "model_id" not in base_sum.columns:
        raise SystemExit(f"[error] base summary missing model_id col: {base_sum_path}")

    model_id = str(base_sum["model_id"].dropna().iloc[0])
    subject_layer = int(base_sum["subject_layer"].dropna().iloc[0])
    object_layer  = int(base_sum["object_layer"].dropna().iloc[0])

    rels = load_relset(args.relation_set)
    if not rels:
        raise SystemExit("[error] empty relation set")

    # load prompts index (id -> record), only keep needed relations
    relset = set(rels)
    id2rec: Dict[str, Dict[str, Any]] = {}
    n_prompt = 0
    n_kept = 0
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_prompt += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rel = rec.get("relation_key") or rec.get("relation") or rec.get("task") or ""
            if not isinstance(rel, str) or rel.strip() not in relset:
                continue
            ex_id = _norm_id(rec.get("id") or rec.get("example_id") or rec.get("uid") or rec.get("idx"))
            if ex_id is None:
                continue
            id2rec[str(ex_id)] = rec
            n_kept += 1

    print(f"[load] prompts: read={n_prompt}, kept(relset)={n_kept}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[load] model_key={args.model_key}", file=sys.stderr)
    print(f"[load] model_id={model_id}", file=sys.stderr)
    print(f"[load] device={device} dtype={torch_dtype}", file=sys.stderr)
    print(f"[load] subject_layer={subject_layer} object_layer={object_layer}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    model.eval()

    # decide chat-template mode
    if args.chat_template == "on":
        use_chat = True
    elif args.chat_template == "off":
        use_chat = False
    else:
        use_chat = detect_chat_template_mode(
            model=model,
            tokenizer=tokenizer,
            id2rec=id2rec,
            base_step5_dir=args.base_step5_dir,
            rels=rels,
            system_prompt=args.system_prompt,
            subject_layer=subject_layer,
            object_layer=object_layer,
            device=device,
            batch_size=args.batch_size,
        )

    # per-relation summary
    rows = []
    skipped_total = 0

    for rel in rels:
        base_rel_path = os.path.join(args.base_step5_dir, "per_relation", f"{rel}.csv.gz")
        if not os.path.exists(base_rel_path):
            print(f"[warn] missing base per_relation for {rel}: {base_rel_path}", file=sys.stderr)
            continue

        base_df = pd.read_csv(base_rel_path, compression="gzip")
        if base_df.empty:
            continue

        # Prepare ordered lists matching base_df order
        prompts, subjects, golds = [], [], []
        splits = []
        stored_base = []
        meta_group = ""
        meta_name = ""

        for _, r in base_df.iterrows():
            ex_id = _norm_id(r.get("id"))
            if ex_id is None:
                continue
            rec = id2rec.get(str(ex_id))
            if rec is None:
                continue

            q = _choose_first_str(rec, QUESTION_KEYS) or ""
            subj = str(rec.get("subject") or "")
            gold = _choose_first_str(rec, GOLD_KEYS) or ""

            if not q or not subj or not gold:
                continue

            prompts.append(q)
            subjects.append(subj)
            golds.append(gold)
            splits.append(str(r.get("split", "")))
            stored_base.append(float(r.get("base_cos", np.nan)))

            if not meta_group:
                meta_group = str(rec.get("relation_group") or "")
            if not meta_name:
                meta_name = str(rec.get("relation_name") or "")

        n_total = int(len(base_df))
        n_used_target = int(len(prompts))

        if n_used_target < 4:
            continue

        # Extract vectors (batched)
        S_list: List[np.ndarray] = []
        O_list: List[np.ndarray] = []
        kept_splits: List[str] = []
        kept_base: List[float] = []

        bs = max(1, int(args.batch_size))
        skipped_align = 0

        for start in range(0, n_used_target, bs):
            batch_prompts = prompts[start:start+bs]
            batch_subjects = subjects[start:start+bs]
            batch_golds = golds[start:start+bs]
            batch_splits = splits[start:start+bs]
            batch_base = stored_base[start:start+bs]

            s_vecs, o_vecs, kept = get_subject_object_vecs_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                subjects=batch_subjects,
                golds=batch_golds,
                system_prompt=args.system_prompt,
                subject_layer=subject_layer,
                object_layer=object_layer,
                device=device,
                use_chat_template=use_chat,
            )
            skipped_align += (len(batch_prompts) - len(kept))

            # kept are local indices into this batch
            for local_i, sv, ov in zip(kept, s_vecs, o_vecs):
                S_list.append(sv.numpy().astype(np.float32))
                O_list.append(ov.numpy().astype(np.float32))
                kept_splits.append(batch_splits[local_i])
                kept_base.append(float(batch_base[local_i]))

        skipped_total += skipped_align

        if len(S_list) < 4:
            continue

        S = np.stack(S_list, axis=0)
        O = np.stack(O_list, axis=0)
        kept_splits_arr = np.array(kept_splits, dtype=object)

        is_train = kept_splits_arr == "train"
        is_test = ~is_train

        n_train = int(is_train.sum())
        n_test = int(is_test.sum())
        if n_train < 2 or n_test < 1:
            continue

        Xtr, Ytr = S[is_train], O[is_train]
        Xte, Yte = S[is_test], O[is_test]

        # base cos on test (recomputed; should match stored base_cos closely)
        base_cos_test = cosine_rows(Xte, Yte)

        # affine prediction via ridge
        try:
            Yhat = ridge_affine_predict_dual(Xtr, Ytr, Xte, lam=float(args.ridge_lambda))
        except Exception as e:
            print(f"[warn] {args.model_key} {rel}: ridge solve failed: {e}", file=sys.stderr)
            continue

        cos_aff_test = cosine_rows(Yhat, Yte)
        delta_aff_test = cos_aff_test - base_cos_test

        row = {
            "model_key": args.model_key,
            "model_id": model_id,
            "relation_key": rel,
            "relation_group": meta_group,
            "relation_name": meta_name,
            "n_total_base": n_total,
            "n_used_after_align": int(len(S_list)),
            "n_train": n_train,
            "n_test": n_test,
            "base_cos_mean_test": float(np.mean(base_cos_test)),
            "affine_cos_mean_test": float(np.mean(cos_aff_test)),
            "cos_improvement_affine": float(np.mean(delta_aff_test)),
            "ridge_lambda": float(args.ridge_lambda),
            "subject_layer": subject_layer,
            "object_layer": object_layer,
            "use_chat_template": bool(use_chat),
            "skipped_align": int(skipped_align),
        }
        rows.append(row)

        print(f"[rel] {args.model_key} {rel}: n_test={n_test} Δcos_aff={row['cos_improvement_affine']:.6f} (skipped_align={skipped_align})", file=sys.stderr)

    out_path = os.path.join(args.outdir, "relation_summary_affine.csv.gz")
    pd.DataFrame(rows).to_csv(out_path, index=False, compression="gzip")
    print(f"[done] wrote: {out_path} rows={len(rows)} skipped_align_total={skipped_total}", file=sys.stderr)


if __name__ == "__main__":
    main()
