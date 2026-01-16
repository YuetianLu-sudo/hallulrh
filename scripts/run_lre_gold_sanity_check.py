#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold-verifiable sanity check on *natural* LRE triples:

For each model and relation, we:
  1) Render each (relation, subject) into the SAME question template as behavioral eval (instruction setting),
     rather than using Hernandez's cloze-style `text`.
  2) Generate an answer under greedy decoding.
  3) Label the output as REFUSAL vs VALUE-PROVIDING using the SAME regex baseline used elsewhere in the paper
     (i.e., identical patterns to scripts/compute_rule_judge_metrics.py).
  4) For VALUE-PROVIDING outputs, check correctness against the gold `answer` from the LRE triple.

We then join with the already-computed relation-level Δcos (cos_improvement) from behavior_plus_lre.csv and
report correlations between Δcos and (value rate, wrong rate, accuracy|value) within each model.

This script is dependency-light:
  - transformers, torch, pandas, numpy, scipy

Example:
  python scripts/run_lre_gold_sanity_check.py \
    --behavior_plus_lre analysis/.../behavior_plus_lre.csv \
    --lre_triples data/lre/natural_relations_ext6_q_fname.jsonl \
    --out_dir analysis/.../gold_sanity_check \
    --batch_size 16 --max_new_tokens 64
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Question templates
# -----------------------------
DEFAULT_TEMPLATES: Dict[str, str] = {
    "father": "What is {subject}'s father's first name?",
    "instrument": "Which instrument did {subject} play?",
    "sport": "Which sport did {subject} play?",
    "company_ceo": "Who is the CEO of {subject}?",
    "company_hq": "Where is {subject} headquartered?",
    "country_language": "What is the official language of {subject}?",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer with a single short phrase."

# -----------------------------
# Refusal regex baseline (IDENTICAL to compute_rule_judge_metrics.py patterns)
# -----------------------------
LABELS = ["REFUSAL", "HALLUCINATION"]  # keep legacy naming; here HALLUCINATION == value-providing


def normalize_text(s: str) -> str:
    """Normalize unicode punctuation to make regex matching stable."""
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


# A compact but high-coverage set of refusal templates. (copied verbatim from your baseline)
DEFAULT_REFUSAL_PATTERNS: List[str] = [
    r"\bi (do not|don't) know\b",
    r"\bi (do not|don't) have\b.*\binformation\b",
    r"\bi (do not|don't) have\b.*\bdata\b",
    r"\bi (do not|don't) have\b.*\baccess\b",
    r"\bi (cannot|can't|could not|couldn't|unable to)\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bcan't\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bcannot\b.*\b(answer|provide|determine|find|locate)\b",
    r"\bnot enough (information|context)\b",
    r"\binsufficient (information|context)\b",
    r"\bwithout (additional|more) information\b",
    r"\bno (reliable )?(information|data)\b",
    r"\bno\b.*\bavailable\b",
    r"\bnot\b.*\bavailable\b",
    r"\bnot provided\b",
    r"\bnot specified\b",
    r"\bdoes not specify\b",
    r"\bnot given\b",
    r"\bunknown\b",
    r"\bnot known\b",
    r"\bunsure\b",
    r"\bnot sure\b",
    r"\bi am not sure\b",
    r"\bi'm not sure\b",
    r"\bi am unsure\b",
    r"\bi'm unsure\b",
    r"\bi am not aware\b",
    r"\bi'm not aware\b",
    r"\bnot aware of\b",
    r"\bthe text (provided )?(does not|doesn't) (specify|reveal|provide|mention|state)\b",
    r"\bthe information (provided )?(does not|doesn't) (specify|reveal|provide|mention|state)\b",
    r"\bthe question (cannot|can't) be answered\b",
    r"\bcannot be answered\b",
    r"\bcan't be answered\b",
    r"\bno official\b.*\bspecified\b",
    r"\bnot (able|authorized) to\b",
    r"\bprivacy\b",
    r"\bsensitive\b",
    r"\bas an ai\b",
    r"\bi apologize\b",
    r"\bi'm sorry\b",
    r"\bi am sorry\b",
    r"\bfictional\b",
    r"\bdoes not exist\b",
    r"\bcan't browse\b",
    r"\bcannot browse\b",
    r"\bno access to\b.*\b(internet|web|database)\b",
]


def build_refusal_regex(patterns: List[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), flags=re.IGNORECASE)


REFUSAL_RE = build_refusal_regex(DEFAULT_REFUSAL_PATTERNS)


def rule_label(answer: str) -> str:
    a = normalize_text(answer).strip()
    if a == "":
        return "REFUSAL"
    if re.fullmatch(r"(?i)(n/?a|none|null|unknown|not applicable)\.?", a):
        return "REFUSAL"
    if REFUSAL_RE.search(a):
        return "REFUSAL"
    if re.fullmatch(r"(?i)(i am (an )?ai language model\.?|i am a helpful assistant\.?)", a):
        return "REFUSAL"
    return "HALLUCINATION"  # value-providing


# -----------------------------
# Simple correctness matching
# -----------------------------
_PUNCT_STRIP = re.compile(r"^[\s\.\,\;\:\!\?\(\)\[\]\{\}\"\'“”‘’`]+|[\s\.\,\;\:\!\?\(\)\[\]\{\}\"\'“”‘’`]+$")


def normalize_for_match(s: str) -> str:
    s = normalize_text(s)
    s = s.strip()
    s = _PUNCT_STRIP.sub("", s)
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def is_correct(pred: str, gold: str) -> bool:
    """
    Conservative correctness:
      - normalize both
      - accept exact match
      - accept if one is a strict substring of the other with word boundaries
    This is designed for "single short phrase" outputs, but robust to minor punctuation.
    """
    p = normalize_for_match(pred)
    g = normalize_for_match(gold)
    if p == "" or g == "":
        return False
    if p == g:
        return True
    # bounded substring match
    if re.search(rf"(?:^|\b){re.escape(g)}(?:\b|$)", p):
        return True
    if re.search(rf"(?:^|\b){re.escape(p)}(?:\b|$)", g):
        return True
    return False


# -----------------------------
# Data structures / IO
# -----------------------------
@dataclass
class Triple:
    relation: str
    subject: str
    answer: str


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_triples(
    paths: List[str],
    relation_key: str,
    subject_key: str,
    answer_key: str,
    restrict_relations: Optional[set] = None,
    max_per_relation: Optional[int] = None,
    seed: int = 0,
) -> List[Triple]:
    rng = random.Random(seed)
    by_rel: Dict[str, List[Triple]] = {}
    for p in paths:
        for row in read_jsonl(p):
            if relation_key not in row or subject_key not in row or answer_key not in row:
                continue
            rel = str(row[relation_key])
            if restrict_relations is not None and rel not in restrict_relations:
                continue
            subj = str(row[subject_key])
            ans = str(row[answer_key])
            by_rel.setdefault(rel, []).append(Triple(rel, subj, ans))

    out: List[Triple] = []
    for rel, items in sorted(by_rel.items()):
        if max_per_relation is not None and len(items) > max_per_relation:
            rng.shuffle(items)
            items = items[:max_per_relation]
        out.extend(items)
    return out


def batched(xs: List[str], bs: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


# -----------------------------
# Prompt rendering and generation
# -----------------------------
def apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Render a single-turn chat prompt using the model's official chat template.
    If the tokenizer doesn't accept a system role, merge system into user.
    Returns a STRING (tokenize=False), so we can batch-tokenize with padding safely.
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        merged = f"{system_prompt}\n\n{user_prompt}"
        msgs2 = [{"role": "user", "content": merged}]
        try:
            return tokenizer.apply_chat_template(
                msgs2, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Last-resort fallback: plain concatenation
            return merged


@torch.no_grad()
def generate_answers(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    device = model.device
    tokenizer.padding_side = "left"

    # Critical fix: many decoder-only tokenizers (Llama) have no pad_token by default.
    # For batched padding we must define one; best practice is pad_token := eos_token.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            tok = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
            tokenizer.pad_token = tok
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    outs: List[str] = []
    for batch_prompts in batched(prompts, batch_size):
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Slice off the prompt part
        prompt_len = input_ids.shape[1]
        for i in range(gen.shape[0]):
            new_tokens = gen[i, prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = text.strip()
            if "\n" in text:
                text = text.splitlines()[0].strip()
            outs.append(text)

    assert len(outs) == len(prompts)
    return outs


# -----------------------------
# Statistics
# -----------------------------
def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import pearsonr
    if len(x) < 3 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import spearmanr
    if len(x) < 3 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior_plus_lre", required=True, help="CSV with columns incl. model_key,relation,cos_improvement,n_test,model_id")
    ap.add_argument("--lre_triples", required=True, help="JSONL path or glob for natural LRE triples (gold answers).")
    ap.add_argument("--out_dir", required=True, help="Output directory to write CSV + logs.")
    ap.add_argument("--models", default="gemma_7b_it,llama3_1_8b_instruct,mistral_7b_instruct,qwen2_5_7b_instruct",
                    help="Comma-separated model_keys to run.")
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_per_relation", type=int, default=1000, help="Cap #triples per relation (for speed).")
    ap.add_argument("--relation_key", default="relation")
    ap.add_argument("--subject_key", default="subject")
    ap.add_argument("--answer_key", default="answer")
    ap.add_argument("--templates_json", default="", help="Optional JSON mapping relation->template.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load behavior_plus_lre and keep unique per model_key x relation
    df = pd.read_csv(args.behavior_plus_lre)
    needed_cols = {"model_key", "relation", "cos_improvement", "model_id"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"behavior_plus_lre is missing required columns: {missing}")

    if "n_test" not in df.columns:
        df["n_test"] = np.nan

    df = df.copy()
    df["relation"] = df["relation"].astype(str)

    # Templates
    templates = dict(DEFAULT_TEMPLATES)
    if args.templates_json:
        with open(args.templates_json, "r", encoding="utf-8") as f:
            templates.update(json.load(f))

    # Map model_key -> model_id (HF checkpoint)
    model_id_map = dict(df.groupby("model_key")["model_id"].first())

    # Determine relations to include: intersection of templates and df relations and triples.
    target_relations = sorted(set(df["relation"].unique()).intersection(set(templates.keys())))
    if not target_relations:
        raise RuntimeError("No overlapping relations found between behavior_plus_lre.csv and templates.")

    # Load triples
    triple_paths = sorted(list(map(str, Path().glob(args.lre_triples))) if any(ch in args.lre_triples for ch in "*?[]") else [args.lre_triples])
    if not triple_paths:
        # Try glob manually
        import glob
        triple_paths = glob.glob(args.lre_triples)
    if not triple_paths:
        raise RuntimeError(f"No files matched --lre_triples {args.lre_triples}")

    triples = load_triples(
        triple_paths,
        relation_key=args.relation_key,
        subject_key=args.subject_key,
        answer_key=args.answer_key,
        restrict_relations=set(target_relations),
        max_per_relation=args.max_per_relation,
        seed=args.seed,
    )
    if not triples:
        raise RuntimeError("Loaded 0 triples. Check keys and file format.")

    # Organize triples by relation
    by_rel: Dict[str, List[Triple]] = {r: [] for r in target_relations}
    for t in triples:
        if t.relation in by_rel:
            by_rel[t.relation].append(t)

    # Prepare output rows
    rows: List[dict] = []

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in model_id_map:
            print(f"[WARN] model_key {mk} not found in behavior_plus_lre.csv; skipping.", file=sys.stderr)
            continue
        model_id = model_id_map[mk]
        print(f"\n== Loading model: {mk} -> {model_id} ==")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        mdl.eval()

        # Run per relation
        for rel in target_relations:
            items = by_rel.get(rel, [])
            if not items:
                continue
            # Render prompts
            tmpl = templates[rel]
            user_prompts = [tmpl.format(subject=t.subject) for t in items]
            chat_prompts = [apply_chat_template(tok, args.system_prompt, up) for up in user_prompts]

            # Generate
            preds = generate_answers(
                mdl, tok, chat_prompts, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
            )

            # Score
            n = len(items)
            n_ref = 0
            n_val = 0
            n_correct = 0
            n_wrong = 0

            for t, pred in zip(items, preds):
                lab = rule_label(pred)
                if lab == "REFUSAL":
                    n_ref += 1
                else:
                    n_val += 1
                    if is_correct(pred, t.answer):
                        n_correct += 1
                    else:
                        n_wrong += 1

            refusal_rate = n_ref / n
            value_rate = n_val / n
            correct_rate = n_correct / n
            wrong_rate = n_wrong / n
            acc_given_value = (n_correct / n_val) if n_val > 0 else float("nan")

            print(
                f"  {rel:15s} n={n:4d} refusal={refusal_rate:.3f} value={value_rate:.3f} "
                f"correct={correct_rate:.3f} wrong={wrong_rate:.3f} acc|value={acc_given_value:.3f}"
            )

            # Join Δcos / n_test from behavior_plus_lre.csv (unique row per mk-rel)
            sub = df[(df["model_key"] == mk) & (df["relation"] == rel)]
            if len(sub) == 0:
                dcos = float("nan")
                n_test = float("nan")
            else:
                dcos = float(sub.iloc[0]["cos_improvement"])
                n_test = float(sub.iloc[0]["n_test"]) if "n_test" in sub.columns else float("nan")

            rows.append(
                dict(
                    model_key=mk,
                    model_id=model_id,
                    relation=rel,
                    n=n,
                    refusal_rate=refusal_rate,
                    value_rate=value_rate,
                    correct_rate=correct_rate,
                    wrong_rate=wrong_rate,
                    acc_given_value=acc_given_value,
                    delta_cos=dcos,
                    n_test=n_test,
                )
            )

        # Free GPU memory between models
        try:
            del mdl
            torch.cuda.empty_cache()
        except Exception:
            pass

    df_out = pd.DataFrame(rows)
    out_csv = out_dir / "gold_sanity_check_by_relation.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

    # Correlations within each model: Δcos vs value_rate / wrong_rate / acc_given_value
    print("\n== Correlations (within each model; n = #relations) ==")
    corr_rows = []
    for mk in sorted(df_out["model_key"].unique()):
        d = df_out[df_out["model_key"] == mk].dropna(subset=["delta_cos"])
        if len(d) < 3:
            continue
        x = d["delta_cos"].to_numpy()
        y_val = d["value_rate"].to_numpy()
        y_wrong = d["wrong_rate"].to_numpy()
        y_acc = d["acc_given_value"].to_numpy()

        r_val, p_val = pearsonr_safe(x, y_val)
        r_wrong, p_wrong = pearsonr_safe(x, y_wrong)
        r_acc, p_acc = pearsonr_safe(x, y_acc)

        print(f"{mk}:  Δcos~value r={r_val:.3f} (p={p_val:.3g}) | Δcos~wrong r={r_wrong:.3f} (p={p_wrong:.3g}) | Δcos~acc|value r={r_acc:.3f} (p={p_acc:.3g})")
        corr_rows.append(dict(model_key=mk, r_val=r_val, p_val=p_val, r_wrong=r_wrong, p_wrong=p_wrong, r_acc=r_acc, p_acc=p_acc, n=len(d)))

    df_corr = pd.DataFrame(corr_rows)
    out_corr = out_dir / "gold_sanity_check_corr.csv"
    df_corr.to_csv(out_corr, index=False)
    print(f"Wrote: {out_corr}")

    # LaTeX snippet (optional convenience)
    print("\n== LaTeX snippet (per-model rows) ==")
    for _, row in df_corr.iterrows():
        mk = row["model_key"]
        print(f"{mk} & {row['r_val']:.3f} & {row['r_wrong']:.3f} & {row['r_acc']:.3f} \\\\")


if __name__ == "__main__":
    main()
