#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute "synthetic-output LRE" Δcos on value-providing outputs.

Goal
----
Address the "distribution mismatch" concern by computing an LRE-style linearity proxy
directly on the SAME synthetic prompts used for behavioral evaluation, using the model's
OWN generated value as the "object" string.

Key design choices (defaults are recommended)
---------------------------------------------
1) Filtering to value-providing vs refusal uses *Gemini* labels already present in
   `all_with_judge_merged.csv` (perfectly consistent with the main paper's behavior rates),
   WITHOUT making any new external judge calls.
2) For reviewer-proofing, we ALSO compute the *same* deterministic regex baseline
   used in Appendix "Rule-based judge baseline" and print agreement stats
   (accuracy + Cohen's kappa). You can optionally switch filtering to regex via
   `--label_source regex` to make this experiment judge-free.

Outputs
-------
- synthetic_lre_by_relation.csv : per (model, relation) value/refusal rates + synthetic Δcos
- synthetic_lre_corr.csv        : per-model correlations between synthetic Δcos and value rate
- merged_with_behavior_plus_lre.csv (optional) : merges in your existing behavior_plus_lre summary

Example
-------
python scripts/compute_synthetic_lre_from_outputs.py \
  --input_csv analysis/chat_neutral_full_20251224_010241_post_v2/all_with_judge_merged.csv \
  --out_dir   analysis/chat_neutral_full_20251224_010241_post_v2/synthetic_lre_from_outputs \
  --batch_size 8 \
  --max_length 256 \
  --min_pairs 50 \
  --seed 42
"""

import argparse
import io
import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

try:
    import scipy.stats as stats  # type: ignore
except Exception:
    stats = None  # type: ignore


# -------------------------
# Canonical model IDs
# -------------------------
DEFAULT_MODEL_ID: Dict[str, str] = {
    "gemma_7b_it": "google/gemma-7b-it",
    "llama3_1_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2_5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
}

MODEL_DISPLAY: Dict[str, str] = {
    "gemma_7b_it": "Gemma-7B-IT",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "mistral_7b_instruct": "Mistral-7B-Instruct",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
}


# -------------------------
# Question templates (must match your synthetic prompts)
# -------------------------
DEFAULT_TEMPLATES: Dict[str, str] = {
    "father":            "What is {subject}'s father's first name?",
    "instrument":        "Which instrument did {subject} play?",
    "sport":             "Which sport did {subject} play?",
    "company_ceo":       "Who is the CEO of {subject}?",
    "company_hq":        "Where is {subject} headquartered?",
    "country_language":  "What is the official language of {subject}?",
}


# -------------------------
# EXACT regex baseline from your compute_rule_judge_metrics.py (Appendix D)
# -------------------------
LABELS = ["REFUSAL", "HALLUCINATION"]

REFUSAL_PATTERNS: List[str] = [
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
REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in REFUSAL_PATTERNS), flags=re.IGNORECASE)


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


def rule_label(answer: str) -> str:
    a = normalize_text(answer).strip()
    if a == "":
        return "REFUSAL"
    # Common placeholders
    if re.fullmatch(r"(?i)(n/?a|none|null|unknown|not applicable)\.?", a):
        return "REFUSAL"
    # Template matching
    if REFUSAL_RE.search(a):
        return "REFUSAL"
    # Generic meta answers
    if re.fullmatch(r"(?i)(i am (an )?ai language model\.?|i am a helpful assistant\.?)", a):
        return "REFUSAL"
    return "HALLUCINATION"


def read_csv_safely(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


def canon_label(x: str) -> str:
    s = str(x).strip().upper()
    if s in {"REFUSAL", "ABSTAIN", "ABSTENTION", "0"}:
        return "REFUSAL"
    if s in {"HALLUCINATION", "VALUE", "VALUE_PROVIDING", "VALUE-PROVIDING", "1"}:
        return "HALLUCINATION"
    if "REFUS" in s:
        return "REFUSAL"
    if "HALL" in s or "VALUE" in s:
        return "HALLUCINATION"
    return s


def canon_relation(x: str) -> str:
    return str(x).strip().lower()


def canon_model_key(x: str) -> str:
    s = str(x).strip()
    if s in DEFAULT_MODEL_ID:
        return s
    s_low = s.lower()
    s_low = s_low.replace("meta-llama/", "").replace("meta_llama/", "")
    s_low = s_low.replace("mistralai/", "").replace("google/", "")
    s_low = s_low.replace("qwen/", "").replace("qwen2.5/", "")
    s_norm = re.sub(r"[^a-z0-9\.]+", "_", s_low).strip("_")

    if "gemma" in s_norm:
        return "gemma_7b_it"
    if "llama" in s_norm and ("3_1" in s_norm or "3.1" in s_norm) and "8b" in s_norm:
        return "llama3_1_8b_instruct"
    if "mistral" in s_norm and "7b" in s_norm:
        return "mistral_7b_instruct"
    if "qwen" in s_norm and ("2_5" in s_norm or "2.5" in s_norm) and "7b" in s_norm:
        return "qwen2_5_7b_instruct"

    if s_norm in DEFAULT_MODEL_ID:
        return s_norm
    return s_norm


def _compile_template_regex(tmpl: str) -> re.Pattern:
    if "{subject}" not in tmpl:
        raise ValueError(f"Template missing {{subject}}: {tmpl}")
    pre, post = tmpl.split("{subject}")
    pat = "^" + re.escape(pre) + r"\s*(?P<subject>.+?)\s*" + re.escape(post) + "$"
    return re.compile(pat)


_TEMPLATE_RE_CACHE: Dict[str, re.Pattern] = {}


def extract_subject(question: str, relation: str, templates: Dict[str, str]) -> Optional[str]:
    q = str(question).strip()
    r = canon_relation(relation)
    tmpl = templates.get(r)
    if tmpl:
        if r not in _TEMPLATE_RE_CACHE:
            _TEMPLATE_RE_CACHE[r] = _compile_template_regex(tmpl)
        m = _TEMPLATE_RE_CACHE[r].match(q)
        if m:
            return m.group("subject").strip()

    # Heuristic fallbacks
    if r == "father":
        m = re.match(r"^What is\s+(.+?)'s father", q)
        if m:
            return m.group(1).strip()
    if r in {"instrument", "sport"}:
        m = re.match(r"^Which\s+(?:instrument|sport)\s+did\s+(.+?)\s+play\??$", q)
        if m:
            return m.group(1).strip()
    if r == "company_ceo":
        m = re.match(r"^Who is the CEO of\s+(.+?)\??$", q)
        if m:
            return m.group(1).strip()
    if r == "company_hq":
        m = re.match(r"^Where is\s+(.+?)\s+headquartered\??$", q)
        if m:
            return m.group(1).strip()
    if r == "country_language":
        m = re.match(r"^What is the official language of\s+(.+?)\??$", q)
        if m:
            return m.group(1).strip()
    return None


def clean_answer(ans: str) -> str:
    a = normalize_text(ans).strip()
    if "\n" in a:
        parts = [p.strip() for p in a.splitlines() if p.strip() != ""]
        a = parts[0] if parts else ""
    return a.strip()


def find_char_span(haystack: str, needle: str, which: str = "first") -> Optional[Tuple[int, int]]:
    if needle == "":
        return None
    i = haystack.find(needle) if which == "first" else haystack.rfind(needle)
    if i < 0:
        return None
    return (i, i + len(needle))


def token_indices_for_span(
    offsets: List[Tuple[int, int]],
    span: Tuple[int, int],
    attention_mask: Optional[List[int]] = None,
) -> List[int]:
    a0, b0 = span
    idx: List[int] = []
    for t, (a, b) in enumerate(offsets):
        if attention_mask is not None and attention_mask[t] == 0:
            continue
        if a == 0 and b == 0:
            continue
        if b > a0 and a < b0:
            idx.append(t)
    return idx


def mean_pool(h: torch.Tensor, idx: List[int]) -> torch.Tensor:
    return h[idx].mean(dim=0)


def cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = (u * v).sum(dim=-1)
    den = (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(eps)
    return num / den


@dataclass
class PairExample:
    full_text: str
    subject: str
    answer: str


@torch.no_grad()
def extract_pairs_representations(
    model: torch.nn.Module,
    tokenizer,
    pairs: List[PairExample],
    subject_layer_block: int,
    object_layer_block: int,
    batch_size: int,
    max_length: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pairs) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device

    S_list: List[np.ndarray] = []
    O_list: List[np.ndarray] = []

    rng = range(0, len(pairs), batch_size)
    it = tqdm(rng, desc="  extracting reps", leave=False) if (tqdm and verbose) else rng

    for start in it:
        batch = pairs[start:start + batch_size]
        texts = [ex.full_text for ex in batch]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        offsets = enc.pop("offset_mapping")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = out.hidden_states
        hs_sub = hidden_states[subject_layer_block + 1]
        hs_obj = hidden_states[object_layer_block + 1]

        attn_cpu: Optional[List[List[int]]] = None
        if attention_mask is not None:
            attn_cpu = attention_mask.detach().cpu().int().tolist()

        for i, ex in enumerate(batch):
            text_i = ex.full_text
            subj = ex.subject
            ans = ex.answer

            subj_span = find_char_span(text_i, subj, which="first")
            obj_span = find_char_span(text_i, ans, which="last")
            if subj_span is None or obj_span is None:
                continue

            offsets_i = [(int(a), int(b)) for (a, b) in offsets[i].tolist()]
            am_i = attn_cpu[i] if attn_cpu is not None else None

            subj_idx = token_indices_for_span(offsets_i, subj_span, am_i)
            obj_idx = token_indices_for_span(offsets_i, obj_span, am_i)
            if len(subj_idx) == 0 or len(obj_idx) == 0:
                continue

            s_vec = mean_pool(hs_sub[i].float().cpu(), subj_idx)
            o_vec = mean_pool(hs_obj[i].float().cpu(), obj_idx)

            S_list.append(s_vec.numpy())
            O_list.append(o_vec.numpy())

        del out, hidden_states, hs_sub, hs_obj, input_ids, attention_mask, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(S_list) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    S = np.stack(S_list, axis=0).astype(np.float32)
    O = np.stack(O_list, axis=0).astype(np.float32)
    return S, O


def compute_delta_cos(S: np.ndarray, O: np.ndarray, seed: int, train_frac: float = 0.75):
    n = int(S.shape[0])
    if n == 0:
        return (float("nan"), 0, 0, float("nan"), float("nan"))

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(math.floor(train_frac * n))
    n_train = min(max(n_train, 1), n - 1)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    S_tr = torch.from_numpy(S[train_idx])
    O_tr = torch.from_numpy(O[train_idx])
    S_te = torch.from_numpy(S[test_idx])
    O_te = torch.from_numpy(O[test_idx])

    dbar = (O_tr - S_tr).mean(dim=0)
    O_hat = S_te + dbar

    cos_base = cosine(S_te, O_te)
    cos_lre = cosine(O_hat, O_te)
    delta = (cos_lre - cos_base).mean().item()

    return (float(delta), int(n), int(len(test_idx)), float(cos_lre.mean().item()), float(cos_base.mean().item()))


def pearsonr_safe(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))
    if stats is None:
        r = float(np.corrcoef(x, y)[0, 1])
        return (r, float("nan"))
    r, p = stats.pearsonr(x, y)
    return (float(r), float(p))


def spearmanr_safe(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))
    if stats is None:
        return (float("nan"), float("nan"))
    rho, p = stats.spearmanr(x, y)
    return (float(rho), float(p))


def confusion_counts(y_true: List[str], y_pred: List[str], labels: List[str]):
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in labels or p not in labels:
            continue
        cm[t][p] += 1
    return cm


def accuracy_from_cm(cm, labels):
    total = sum(cm[t][p] for t in labels for p in labels)
    correct = sum(cm[l][l] for l in labels)
    return correct / total if total else float("nan")


def kappa_from_cm(cm, labels):
    total = sum(cm[t][p] for t in labels for p in labels)
    if total == 0:
        return float("nan")
    po = accuracy_from_cm(cm, labels)
    row_marg = {t: sum(cm[t][p] for p in labels) for t in labels}
    col_marg = {p: sum(cm[t][p] for t in labels) for p in labels}
    pe = sum(row_marg[l] * col_marg[l] for l in labels) / (total * total)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_source", choices=["gemini", "regex"], default="gemini",
                    help="Default gemini = consistent with main results; regex = judge-free.")
    ap.add_argument("--models", default="all", help="Comma-separated model keys or 'all'")
    ap.add_argument("--templates_json", default="", help="Optional relation->template JSON with {subject}")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--min_pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--merge_behavior_plus_lre", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    templates = dict(DEFAULT_TEMPLATES)
    if args.templates_json:
        with open(args.templates_json, "r", encoding="utf-8") as f:
            user_t = json.load(f)
        for k, v in user_t.items():
            templates[canon_relation(k)] = str(v)

    df = read_csv_safely(args.input_csv)

    model_col = infer_col(df, ["model_key", "model_name", "model", "model_id"])
    task_col = infer_col(df, ["task", "relation", "rel"])
    question_col = infer_col(df, ["question", "prompt", "query"])
    answer_col = infer_col(df, ["answer", "output", "completion"])
    subj_col = infer_col(df, ["subject", "entity", "name"])  # optional
    judge_col = infer_col(df, ["judge_label", "label", "gemini_label"])

    for name, col in [("model", model_col), ("task", task_col), ("question", question_col), ("answer", answer_col)]:
        if col is None:
            raise ValueError(f"Missing required column for {name}. Found columns={list(df.columns)}")

    if args.label_source == "gemini" and judge_col is None:
        raise ValueError("label_source=gemini but no judge label column found (judge_label/label/gemini_label).")

    df["_model_key"] = df[model_col].apply(canon_model_key)
    df["_relation"] = df[task_col].apply(canon_relation)
    df["_question"] = df[question_col].astype(str)
    df["_answer_raw"] = df[answer_col].astype(str)

    df["_regex_label"] = df["_answer_raw"].apply(rule_label)

    if judge_col is not None:
        df["_gemini_label"] = df[judge_col].apply(canon_label)
        y_true = df["_gemini_label"].tolist()
        y_pred = df["_regex_label"].tolist()
        cm = confusion_counts(y_true, y_pred, LABELS)
        acc = accuracy_from_cm(cm, LABELS)
        kappa = kappa_from_cm(cm, LABELS)
        print("== Agreement: Gemini vs Regex (on input_csv) ==")
        print(f"N = {len(df)}  Acc={acc:.4f}  Kappa={kappa:.4f}")
        print("Confusion (rows=true gemini, cols=regex):")
        print(f"           pred=REFUSAL   pred=HALLUCINATION")
        print(f"true=REFUSAL        {cm['REFUSAL']['REFUSAL']:6d}         {cm['REFUSAL']['HALLUCINATION']:6d}")
        print(f"true=HALLUCINATION  {cm['HALLUCINATION']['REFUSAL']:6d}         {cm['HALLUCINATION']['HALLUCINATION']:6d}")
        print()

    df["_label_use"] = df["_gemini_label"] if args.label_source == "gemini" else df["_regex_label"]

    if args.models.strip().lower() == "all":
        model_keys = list(DEFAULT_MODEL_ID.keys())
    else:
        model_keys = [canon_model_key(x) for x in args.models.split(",") if x.strip() != ""]
        model_keys = [k for k in model_keys if k in DEFAULT_MODEL_ID]
    if not model_keys:
        raise ValueError("No valid models selected.")

    relations = sorted(df["_relation"].unique().tolist())

    out_rows = []

    for mk in model_keys:
        model_id = DEFAULT_MODEL_ID[mk]
        print(f"\n== Loading model: {mk} -> {model_id} ==")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()

        L = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "n_layer", None)
        if L is None:
            raise ValueError(f"Cannot infer num_hidden_layers for model {model_id}")
        subj_layer = int(math.floor(L / 2))
        obj_layer = int(L - 2)
        if args.verbose:
            print(f"  Using layers: subject={subj_layer} object={obj_layer} (L={L})")

        df_m = df[df["_model_key"] == mk]
        if df_m.empty:
            print(f"[WARN] No rows for model_key={mk} in input_csv")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        for rel in relations:
            df_mr = df_m[df_m["_relation"] == rel]
            if df_mr.empty:
                continue

            total_n = int(len(df_mr))
            value_mask = (df_mr["_label_use"] == "HALLUCINATION")
            n_value = int(value_mask.sum())
            n_refusal = total_n - n_value
            value_rate = n_value / total_n if total_n else float("nan")
            refusal_rate = n_refusal / total_n if total_n else float("nan")

            if n_value < args.min_pairs:
                out_rows.append({
                    "model_key": mk, "model_id": model_id, "relation": rel,
                    "total_n": total_n, "n_value": n_value, "n_refusal": n_refusal,
                    "value_rate": value_rate, "refusal_rate": refusal_rate,
                    "n_pairs_retained": 0, "n_test": 0,
                    "delta_cos_synth": float("nan"),
                    "cos_lre_mean": float("nan"), "cos_base_mean": float("nan"),
                    "subject_layer": subj_layer, "object_layer": obj_layer,
                    "label_source": args.label_source,
                })
                if args.verbose:
                    print(f"  {rel:15s} value_n={n_value:4d} < min_pairs({args.min_pairs}), skip LRE")
                continue

            pairs = []
            df_val = df_mr[value_mask].copy()
            for _, row in df_val.iterrows():
                q = str(row["_question"]).strip()
                a_raw = str(row["_answer_raw"])
                a = clean_answer(a_raw)
                if a == "":
                    continue

                subj = None
                if subj_col is not None and pd.notna(row.get(subj_col)):
                    subj = str(row.get(subj_col)).strip()
                if not subj:
                    subj = extract_subject(q, rel, templates)
                if not subj:
                    continue

                full_text = q + " " + a
                pairs.append(PairExample(full_text=full_text, subject=subj, answer=a))

            S, O = extract_pairs_representations(
                model=model, tokenizer=tokenizer, pairs=pairs,
                subject_layer_block=subj_layer, object_layer_block=obj_layer,
                batch_size=args.batch_size, max_length=args.max_length,
                verbose=args.verbose,
            )

            if S.shape[0] < args.min_pairs:
                out_rows.append({
                    "model_key": mk, "model_id": model_id, "relation": rel,
                    "total_n": total_n, "n_value": n_value, "n_refusal": n_refusal,
                    "value_rate": value_rate, "refusal_rate": refusal_rate,
                    "n_pairs_retained": int(S.shape[0]), "n_test": 0,
                    "delta_cos_synth": float("nan"),
                    "cos_lre_mean": float("nan"), "cos_base_mean": float("nan"),
                    "subject_layer": subj_layer, "object_layer": obj_layer,
                    "label_source": args.label_source,
                })
                if args.verbose:
                    print(f"  {rel:15s} retained_pairs={S.shape[0]} < min_pairs, skip Δcos")
                continue

            delta, n_pairs, n_test, cos_lre_mean, cos_base_mean = compute_delta_cos(S, O, seed=args.seed)
            out_rows.append({
                "model_key": mk, "model_id": model_id, "relation": rel,
                "total_n": total_n, "n_value": n_value, "n_refusal": n_refusal,
                "value_rate": value_rate, "refusal_rate": refusal_rate,
                "n_pairs_retained": n_pairs, "n_test": n_test,
                "delta_cos_synth": delta,
                "cos_lre_mean": cos_lre_mean, "cos_base_mean": cos_base_mean,
                "subject_layer": subj_layer, "object_layer": obj_layer,
                "label_source": args.label_source,
            })

            print(f"  {rel:15s} total={total_n:4d} value={n_value:4d} Δcos_synth={delta: .3f} (retained={n_pairs}, test={n_test})")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_out = pd.DataFrame(out_rows)
    out_path = os.path.join(args.out_dir, "synthetic_lre_by_relation.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    corr_rows = []
    for mk in sorted(df_out["model_key"].unique().tolist()):
        sub = df_out[(df_out["model_key"] == mk) & (~pd.isna(df_out["delta_cos_synth"]))].copy()
        x = sub["delta_cos_synth"].to_numpy(dtype=float)
        y = sub["value_rate"].to_numpy(dtype=float)
        r, p = pearsonr_safe(x, y)
        rho, p_s = spearmanr_safe(x, y)
        corr_rows.append({
            "model_key": mk,
            "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": p_s,
            "n_relations": int(len(sub)),
        })

    df_corr = pd.DataFrame(corr_rows)
    corr_path = os.path.join(args.out_dir, "synthetic_lre_corr.csv")
    df_corr.to_csv(corr_path, index=False)
    print(f"Wrote: {corr_path}")

    print("\n== Correlations (within each model; Δcos_synth vs value_rate) ==")
    for _, row in df_corr.iterrows():
        mk = row["model_key"]
        disp = MODEL_DISPLAY.get(mk, mk)
        print(f"{disp:22s}: r={row['pearson_r']:.3f} (p={row['pearson_p']})  "
              f"rho={row['spearman_rho']:.3f} (p={row['spearman_p']})  n={int(row['n_relations'])}")

    print("\n== LaTeX snippet (per-model rows; Δcos_synth vs value_rate) ==")
    for _, row in df_corr.iterrows():
        mk = row["model_key"]
        disp = MODEL_DISPLAY.get(mk, mk).replace("_", r"\_")
        r = row["pearson_r"]
        rho = row["spearman_rho"]
        n = int(row["n_relations"])
        if isinstance(r, float) and not math.isnan(r):
            print(f"{disp} & {r:.3f} & {rho:.3f} & {n} \\\\")
        else:
            print(f"{disp} & -- & -- & {n} \\\\")

    if args.merge_behavior_plus_lre:
        bp = read_csv_safely(args.merge_behavior_plus_lre)
        if "model_key" in bp.columns and "relation" in bp.columns:
            bp["_model_key"] = bp["model_key"].apply(canon_model_key)
            bp["_relation"] = bp["relation"].apply(canon_relation)
            merged = df_out.merge(
                bp,
                left_on=["model_key", "relation"],
                right_on=["_model_key", "_relation"],
                how="left",
                suffixes=("", "_behavior"),
            )
            mpath = os.path.join(args.out_dir, "merged_with_behavior_plus_lre.csv")
            merged.to_csv(mpath, index=False)
            print(f"Wrote: {mpath}")
        else:
            print("[WARN] merge_behavior_plus_lre provided but missing model_key/relation columns; skipped.")

if __name__ == "__main__":
    main()
