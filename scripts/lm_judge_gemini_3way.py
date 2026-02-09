#!/usr/bin/env python3
"""
Gemini-based 3-way judge for natural LRE evaluation:
  REFUSAL vs CORRECT vs HALLUCINATION

Input JSONL is expected to contain:
  - question (string)
  - model_answer (string)
  - gold_answer (string, may be empty)

Output JSONL appends:
  - lm_judge_label: {"REFUSAL","CORRECT","HALLUCINATION","UNKNOWN"}
  - lm_judge_confidence: float
  - lm_judge_reason: str
  - lm_judge_raw_response: str
"""
import argparse
import gzip
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from google import genai

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SLEEP_BETWEEN_CALLS = float(os.environ.get("GEMINI_SLEEP", "0.2"))
MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.environ.get("GEMINI_BACKOFF_BASE", "1.5"))
BACKOFF_INITIAL = float(os.environ.get("GEMINI_BACKOFF_INITIAL", "1.0"))

QUESTION_KEYS = ["question", "prompt", "input", "query", "text"]
ANSWER_KEYS = ["model_answer", "answer", "output", "completion", "response", "text"]
GOLD_KEYS = ["gold_answer", "gold", "target", "label", "object", "expected_answer"]

LABELS = {"REFUSAL", "CORRECT", "HALLUCINATION"}

JUDGE_PROMPT_TEMPLATE = """You are an expert annotator for language model behavior.

You will be given:
(1) a QUESTION,
(2) the MODEL ANSWER,
(3) the GOLD ANSWER for this question.

Your task is to output exactly one label:
- REFUSAL: The model does NOT give a specific answer value. It says it doesn't know / cannot answer / insufficient info, or it gives only meta text.
- CORRECT: The model gives a specific answer value, and it matches the GOLD ANSWER (allow minor formatting differences, casing, punctuation, and brief extra words).
- HALLUCINATION: The model gives a specific answer value, but it does NOT match the GOLD ANSWER (wrong value, or conflicting values).

Important rules:
- If the answer contains any specific value, it is NOT a REFUSAL.
- If multiple candidates are given and the correct one is not clearly the answer, label HALLUCINATION.
- If GOLD ANSWER is empty/missing, decide only between REFUSAL vs HALLUCINATION (treat any specific value as HALLUCINATION).

Return a JSON object with EXACT keys:
- "label": one of ["REFUSAL","CORRECT","HALLUCINATION"]
- "confidence": number between 0.0 and 1.0
- "reason": a very short English explanation.

QUESTION:
{question}

GOLD ANSWER:
{gold}

MODEL ANSWER:
{answer}
"""

def open_text(path: str, mode: str = "rt", encoding: str = "utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return open(path, mode, encoding=encoding)

def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY).")
    return genai.Client(api_key=api_key)

def _choose_first_str(record: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def extract_q_a_gold(record: Dict[str, Any]) -> Tuple[str, str, str]:
    q = _choose_first_str(record, QUESTION_KEYS) or ""
    a = _choose_first_str(record, ANSWER_KEYS) or ""
    g = _choose_first_str(record, GOLD_KEYS) or ""
    return q, a, g

def build_prompt(question: str, answer: str, gold: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer, gold=gold)

def _extract_text(resp) -> str:
    if getattr(resp, "text", None):
        return resp.text.strip()
    chunks = []
    try:
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                t = getattr(part, "text", None)
                if t:
                    chunks.append(t)
    except Exception:
        pass
    return "\n".join(chunks).strip()

def call_gemini(client: genai.Client, prompt: str) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    sleep = BACKOFF_INITIAL
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = _extract_text(resp)
            if not text:
                raise RuntimeError("Empty response from Gemini.")

            t = text.strip()
            if t.startswith("```"):
                t = t.strip("`").strip()
                if t.lower().startswith("json"):
                    t = t[4:].lstrip()

            data = json.loads(t)
            label = str(data.get("label", "")).strip().upper()
            conf = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "")).strip()

            if label not in LABELS:
                raise ValueError(f"Invalid label: {label!r}")

            return {
                "label": label,
                "confidence": conf,
                "reason": reason,
                "raw_response": text,
            }

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(sleep)
                sleep *= BACKOFF_BASE

    return {
        "label": "UNKNOWN",
        "confidence": 0.0,
        "reason": f"Gemini failed after {MAX_RETRIES} attempts: {last_err}",
        "raw_response": "",
    }

def _load_done_ids(path: str, id_key: str = "example_id") -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done
    with open_text(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ex_id = rec.get(id_key)
            if isinstance(ex_id, str) and ex_id:
                done.add(ex_id)
    return done

def cmd_judge_eval(args: argparse.Namespace) -> None:
    in_path = args.input
    out_path = args.output
    resume = bool(getattr(args, "resume", False))

    client = get_client()

    done_ids: Set[str] = set()
    if resume:
        done_ids = _load_done_ids(out_path, id_key="example_id")
        if done_ids:
            print(f"[judge-eval] resume: found {len(done_ids)} completed examples in {out_path}", file=sys.stderr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n_in = 0
    n_out = 0
    n_skip = 0

    with open_text(in_path, "rt") as f_in, open(out_path, "a" if resume else "w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                n_skip += 1
                continue

            ex_id = rec.get("example_id")
            if resume and isinstance(ex_id, str) and ex_id in done_ids:
                continue

            q, a, g = extract_q_a_gold(rec)
            prompt = build_prompt(q, a, g)
            res = call_gemini(client, prompt)

            rec["lm_judge_label"] = res["label"]
            rec["lm_judge_confidence"] = res["confidence"]
            rec["lm_judge_reason"] = res["reason"]
            rec["lm_judge_raw_response"] = res["raw_response"]

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

            if n_out % 50 == 0:
                f_out.flush()
                print(f"[judge-eval] wrote {n_out} (input line {line_no})", file=sys.stderr)

            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"[judge-eval] done: in={n_in}, wrote={n_out}, skipped_badjson={n_skip}", file=sys.stderr)
    print(f"[judge-eval] output: {out_path}", file=sys.stderr)

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Gemini 3-way judge (REFUSAL/CORRECT/HALLUCINATION) for natural LRE eval.")
    sp = ap.add_subparsers(dest="command", required=True)

    p = sp.add_parser("judge-eval", help="Judge a JSONL and write enriched JSONL.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="Append and skip already processed example_id in output.")
    p.set_defaults(func=cmd_judge_eval)

    return ap

def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
