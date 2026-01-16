#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# ------------- Config -------------

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_SAMPLE_SIZE = 200
SLEEP_BETWEEN_CALLS = float(os.environ.get("GEMINI_SLEEP", "0.2"))  # seconds

QUESTION_KEYS = ["question", "prompt", "input", "query"]
ANSWER_KEYS = ["answer", "output", "completion", "response", "model_answer", "text"]

VALID_LABELS = {"refusal": "REFUSAL", "hallucination": "HALLUCINATION"}


# ------------- Gemini client helpers -------------

def get_client() -> genai.Client:
    """Create a Gemini client using GEMINI_API_KEY or GOOGLE_API_KEY."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Please set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.\n"
            "Example: export GEMINI_API_KEY='YOUR_REAL_KEY'"
        )
    client = genai.Client(api_key=api_key)
    return client


def _extract_text_from_response(resp) -> str:
    """Best-effort text extraction from a Gemini response."""
    # Fast path: use the helper if it already contains only text parts.
    if getattr(resp, "text", None):
        return resp.text.strip()

    # Fallback: concatenate all text parts from all candidates.
    chunks = []
    try:
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                piece = getattr(part, "text", None)
                if piece:
                    chunks.append(piece)
    except Exception:
        # Best effort only; if it fails we will just return an empty string.
        pass

    return "\n".join(chunks).strip()


def call_gemini(client, prompt: str, max_retries: int = 3, sleep_sec: float = 1.0):
    """Call Gemini and parse the JSON judge output."""
    last_error = None
    valid_labels = {"REFUSAL", "HALLUCINATION"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            text = _extract_text_from_response(resp)

            if not text:
                raise RuntimeError("Empty response from Gemini (no text content).")

            # Strip code fences if the model wraps JSON in ```...``` blocks.
            if text.startswith("```"):
                text = text.strip()
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].lstrip()

            data = json.loads(text)

            raw_label = str(data.get("label", "")).strip().upper()
            confidence = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", ""))

            if raw_label not in valid_labels:
                raise ValueError(f"Invalid label from model: {raw_label!r}")

            return {
                "label": raw_label,
                "confidence": confidence,
                "reason": reason,
                "raw_response": text,
            }

        except Exception as e:
            # Remember the last error and optionally retry.
            last_error = e
            if attempt < max_retries:
                time.sleep(sleep_sec)

    # If we get here, all attempts failed.
    return {
        "label": "unknown",
        "confidence": 0.0,
        "reason": f"Gemini call failed after {max_retries} attempts: {last_error}",
        "raw_response": "",
    }

# ------------- Data extraction helpers -------------

def _choose_first_str(record: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def extract_qa(record: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Heuristically extract (task, model_name, question, answer) from an eval record."""
    task = (
        str(record.get("task"))
        or str(record.get("property") or record.get("relation") or "")
    )
    model_name = (
        str(record.get("model_name"))
        or str(record.get("model") or record.get("checkpoint") or "")
    )

    question = _choose_first_str(record, QUESTION_KEYS)
    answer = _choose_first_str(record, ANSWER_KEYS)

    if question is None or answer is None:
        raise ValueError("Could not find question/answer fields in record.")

    return task, model_name, question, answer


# ------------- Judge prompt -------------

JUDGE_PROMPT_TEMPLATE = """You are an expert annotator for language model behaviour.

Your task is to label how a model answered a question. You only see the question and the model's answer.
You must decide whether the answer is a REFUSAL or a HALLUCINATION.

Definitions:

- "refusal": The answer clearly states that the information is unknown, unavailable, not provided,
  private, or that the model cannot answer. It does NOT propose any specific value for the requested
  attribute. Phrases like "I do not know", "no public information", "cannot determine", "not specified"
  are typical refusals. Or the answer is off-topic, purely meta (for example explaining its training data or safety
  rules), or otherwise does not fit the hallucination below.

- "hallucination": The answer gives a specific value for the requested attribute (for example a first
  name, an instrument, a city) even if it is uncertain or hedged (e.g. "I am not sure, but his name
  might be John"). Any specific guess for the attribute counts as hallucination, even if it might
  accidentally be correct.

Return a JSON object with EXACT keys:
- "label": one of ["refusal", "hallucination"]
- "confidence": number between 0.0 and 1.0 (your confidence in this label)
- "reason": a very short English explanation.

Question:
{question}

Model answer:
{answer}
"""

def build_judge_prompt(question: str, answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)


# ------------- Commands -------------

def cmd_create_sample(args: argparse.Namespace) -> None:
    """Create a small CSV sample from an eval JSONL file for manual labeling."""
    in_path = args.input
    out_path = args.output
    sample_size = args.sample_size
    seed = args.seed

    rows: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f_in:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                task, model_name, question, answer = extract_qa(record)
                rows.append(
                    {
                        "task": task,
                        "model_name": model_name,
                        "question": question,
                        "answer": answer,
                    }
                )
            except Exception as e:
                print(f"[create-sample] Skipping line {line_no}: {e}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No usable examples found in input JSONL.")

    rng = random.Random(seed)
    rng.shuffle(rows)
    if sample_size > 0:
        rows = rows[:sample_size]

    fieldnames = [
        "sample_id",
        "task",
        "model_name",
        "question",
        "answer",
        "manual_label",
        "judge_label",
        "judge_confidence",
        "judge_reason",
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            row_out = {
                "sample_id": i,
                **row,
                "manual_label": "",
                "judge_label": "",
                "judge_confidence": "",
                "judge_reason": "",
            }
            writer.writerow(row_out)

    print(f"[create-sample] Wrote {len(rows)} examples to {out_path}")


def cmd_judge_csv(args: argparse.Namespace) -> None:
    """Run Gemini judge on a CSV."""
    in_path = args.input
    out_path = args.output
    resume = bool(getattr(args, "resume", False))

    client = get_client()

    with open(in_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        in_rows = list(reader)

    if not in_rows:
        raise RuntimeError(f"No rows found in input CSV: {in_path}")

    # Determine fieldnames (ensure judge columns exist)
    base_fieldnames = reader.fieldnames or []
    fieldnames = list(base_fieldnames)
    for col in ["judge_label", "judge_confidence", "judge_reason"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # Resume: count already written rows in out_path (CSV-aware; safe with quoted commas/newlines)
    start_idx = 0
    if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", encoding="utf-8", newline="") as f_prev:
            prev_reader = csv.DictReader(f_prev)
            prev_rows = list(prev_reader)
            start_idx = len(prev_rows)

        if start_idx >= len(in_rows):
            print(f"[judge-csv] Already complete: {out_path} ({start_idx}/{len(in_rows)})")
            return

        print(f"[judge-csv] Resuming: {out_path} ({start_idx}/{len(in_rows)})")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        f_out = open(out_path, "a", encoding="utf-8", newline="")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        f_out = open(out_path, "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

    try:
        for idx in range(start_idx, len(in_rows)):
            row = in_rows[idx]
            question = row.get("question", "")
            answer = row.get("answer", "")

            prompt = build_judge_prompt(question, answer)
            result = call_gemini(client, prompt)

            row["judge_label"] = result["label"]
            row["judge_confidence"] = f"{result['confidence']:.3f}"
            row["judge_reason"] = result["reason"]
            writer.writerow(row)

            # Flush so we don't lose progress if interrupted.
            if (idx + 1) % 20 == 0:
                f_out.flush()
                print(f"[judge-csv] Processed {idx + 1}/{len(in_rows)} examples...")

            time.sleep(SLEEP_BETWEEN_CALLS)

        f_out.flush()
    finally:
        f_out.close()

    print(f"[judge-csv] Wrote judged CSV to {out_path}")


def cmd_eval_judge(args: argparse.Namespace) -> None:
    """Compare Gemini judge labels with manual labels in a CSV."""
    in_path = args.input

    with open(in_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    labels = ["REFUSAL", "HALLUCINATION", "OTHER"]
    cm: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    total = 0
    correct = 0

    for row in rows:
        gold = (row.get("manual_label") or "").strip().upper()
        pred = (row.get("judge_label") or "").strip().upper()
        if gold not in labels or pred not in labels:
            continue
        cm[gold][pred] += 1
        total += 1
        if gold == pred:
            correct += 1

    if total == 0:
        print("[eval-judge] No comparable labels found.")
        return

    print(f"[eval-judge] Total comparable examples: {total}")
    print(f"[eval-judge] Overall accuracy: {correct / total:.3f}")
    print()
    print("Confusion matrix (rows = gold, cols = predicted):")
    header = ["gold \\ pred"] + labels
    print("\t".join(header))
    for g in labels:
        row_counts = [g] + [str(cm[g][p]) for p in labels]
        print("\t".join(row_counts))

    print()
    print("Per-label precision/recall/F1:")
    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[g][l] for g in labels if g != l)
        fn = sum(cm[l][p] for p in labels if p != l)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        print(
            f"- {l}: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f} "
            f"(tp={tp}, fp={fp}, fn={fn})"
        )


def cmd_judge_eval(args: argparse.Namespace) -> None:
    """Run Gemini judge on a full eval JSONL file and write enriched JSONL."""
    in_path = args.input
    out_path = args.output

    client = get_client()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as f_in, open(
        out_path, "w", encoding="utf-8"
    ) as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                _, _, question, answer = extract_qa(record)
            except Exception as e:
                print(f"[judge-eval] Skipping line {line_no}: {e}", file=sys.stderr)
                continue

            prompt = build_judge_prompt(question, answer)
            result = call_gemini(client, prompt)

            record["lm_judge_label"] = result["label"]
            record["lm_judge_confidence"] = result["confidence"]
            record["lm_judge_reason"] = result["reason"]
            record["lm_judge_raw_response"] = result["raw_response"]

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if line_no % 20 == 0:
                print(f"[judge-eval] Processed {line_no} lines...")
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"[judge-eval] Wrote judged JSONL to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini-based LM-as-judge for refusal vs hallucination."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-sample
    p_sample = subparsers.add_parser(
        "create-sample", help="Create a small CSV sample from eval JSONL for manual labels."
    )
    p_sample.add_argument("--input", required=True, help="Input eval JSONL file.")
    p_sample.add_argument("--output", required=True, help="Output CSV path.")
    p_sample.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of examples to sample (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    p_sample.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    p_sample.set_defaults(func=cmd_create_sample)

    # judge-csv
    p_judge_csv = subparsers.add_parser(
        "judge-csv",
        help="Run Gemini judge on a CSV (e.g. manual calibration sample).",
    )
    p_judge_csv.add_argument("--input", required=True, help="Input CSV path.")
    p_judge_csv.add_argument("--output", required=True, help="Output CSV path.")
    p_judge_csv.add_argument(
    "--resume",
    action="store_true",
    help="Resume from an existing output CSV by appending remaining rows.",
    )
    p_judge_csv.set_defaults(func=cmd_judge_csv)

    # eval-judge
    p_eval = subparsers.add_parser(
        "eval-judge",
        help="Compare Gemini judge labels with manual labels in a CSV.",
    )
    p_eval.add_argument("--input", required=True, help="Input CSV path.")
    p_eval.set_defaults(func=cmd_eval_judge)

    # judge-eval
    p_judge_eval = subparsers.add_parser(
        "judge-eval",
        help="Run Gemini judge on a full eval JSONL and write enriched JSONL.",
    )
    p_judge_eval.add_argument("--input", required=True, help="Input eval JSONL path.")
    p_judge_eval.add_argument(
        "--output", required=True, help="Output enriched JSONL path."
    )
    p_judge_eval.set_defaults(func=cmd_judge_eval)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
