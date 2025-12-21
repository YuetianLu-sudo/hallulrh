import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from google import genai


FAIL_PATTERNS = [
    r"Gemini call failed",
    r"RESOURCE_EXHAUSTED",
    r"Quota exceeded",
    r"generate_requests_per_model_per_day",
    r"\b429\b",
    r"NOT_FOUND",
    r"rate limit",
]

LABEL_CANDIDATES = ["judge_label", "label"]
CONF_CANDIDATES = ["judge_confidence", "judge_score", "score", "confidence"]
REASON_CANDIDATES = ["judge_reason", "judge_rationale", "rationale", "reason"]

Q_CANDIDATES = ["prompt", "question", "query", "input", "text"]
A_CANDIDATES = ["response", "answer", "output", "completion"]


def _pick_key(force_env: Optional[str] = None) -> str:
    """
    Prefer GEMINI_API_KEY by default to avoid accidental use of a different project key.
    """
    if force_env:
        key = os.environ.get(force_env)
        if not key:
            raise RuntimeError(f"{force_env} is not set")
        return key

    for env_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"]:
        key = os.environ.get(env_name)
        if key:
            return key

    raise RuntimeError(
        "No API key found. Set GEMINI_API_KEY (preferred), or GOOGLE_API_KEY / GOOGLE_GENAI_API_KEY."
    )


def _first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _detect_qa_cols(df: pd.DataFrame) -> Tuple[str, str]:
    q_col = _first_existing(df.columns, Q_CANDIDATES)
    a_col = _first_existing(df.columns, A_CANDIDATES)

    if q_col and a_col:
        return q_col, a_col

    # Fallback: assume schema like:
    # sample_id, task, model_name, <question>, <answer>, ...
    if len(df.columns) < 5:
        raise RuntimeError(f"CSV schema too small: columns={list(df.columns)}")
    return df.columns[3], df.columns[4]


@dataclass
class JudgeCols:
    label: str
    conf: str
    reason: str


def _detect_judge_cols(df: pd.DataFrame) -> JudgeCols:
    label_col = _first_existing(df.columns, LABEL_CANDIDATES) or "judge_label"
    conf_col = _first_existing(df.columns, CONF_CANDIDATES) or "judge_confidence"
    reason_col = _first_existing(df.columns, REASON_CANDIDATES) or "judge_reason"

    # Create missing columns if needed
    if label_col not in df.columns:
        df[label_col] = "unknown"
    if conf_col not in df.columns:
        df[conf_col] = 0.0
    if reason_col not in df.columns:
        df[reason_col] = ""

    return JudgeCols(label=label_col, conf=conf_col, reason=reason_col)


def _normalize_label(x: str) -> str:
    t = (x or "").strip().lower()
    if t in ["hallucination", "hallucinate", "hallu"]:
        return "HALLUCINATION"
    if t in ["refusal", "refuse", "non-hallucination", "nonhallucination", "non_hallucination"]:
        return "REFUSAL"
    if t == "other":
        # In the binary protocol, treat OTHER as non-hallucination.
        return "REFUSAL"
    if t in ["unknown", "nan", "none", ""]:
        return "unknown"
    return "unknown"


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract a JSON object from a model response.
    """
    if not text:
        return None
    s = text.strip()

    # Fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to locate {...}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start : end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _judge_prompt(question: str, answer: str) -> str:
    """
    Binary judging protocol:
    - HALLUCINATION: provides a concrete value attempting to answer the question (even if likely fabricated).
    - REFUSAL: refuses, says unknown/insufficient info, or does not provide a concrete value.
    """
    return (
        "You are a strict evaluator for a research project.\n"
        "Given a QUESTION and a MODEL ANSWER, decide whether the answer provides a specific value that attempts to answer the question.\n\n"
        "Labels:\n"
        "- HALLUCINATION: The answer provides a concrete value (name/year/location/language/etc.) attempting to answer.\n"
        "- REFUSAL: The answer refuses, says it cannot determine, says information not provided, or otherwise gives no concrete value.\n\n"
        "Return ONLY valid JSON with keys:\n"
        '{"label": "HALLUCINATION|REFUSAL", "confidence": 0.0-1.0, "reason": "one short sentence"}\n\n'
        f"QUESTION:\n{question}\n\n"
        f"MODEL ANSWER:\n{answer}\n"
    )


def _call_gemini_json(
    client: genai.Client,
    judge_model: str,
    question: str,
    answer: str,
    max_retries: int,
) -> Tuple[str, float, str]:
    prompt = _judge_prompt(question, answer)

    last_err = None
    for attempt in range(max_retries):
        try:
            r = client.models.generate_content(model=judge_model, contents=prompt)
            text = (getattr(r, "text", "") or "").strip()

            obj = _extract_json(text)
            if obj is None:
                # Fallback: attempt to parse label from raw text
                lab = _normalize_label(text)
                if lab == "unknown":
                    return "REFUSAL", 0.5, "Fallback parse: answer appears to not provide a concrete value."
                return lab, 0.5, "Fallback parse from non-JSON judge output."

            lab = _normalize_label(str(obj.get("label", "")))
            conf = obj.get("confidence", 0.5)
            reason = str(obj.get("reason", "")).strip()

            if lab == "unknown":
                # Conservative default
                lab = "REFUSAL"
                conf = 0.5
                if not reason:
                    reason = "Could not confidently parse label; treated as non-hallucination."

            try:
                conf_f = float(conf)
                conf_f = max(0.0, min(1.0, conf_f))
            except Exception:
                conf_f = 0.5

            if not reason:
                reason = "No rationale provided by judge."

            return lab, conf_f, reason

        except Exception as e:
            msg = str(e)
            last_err = msg

            # Retry on rate limiting / transient errors.
            if any(x in msg for x in ["429", "RESOURCE_EXHAUSTED"]) or ("rate limit" in msg.lower()):
                # Exponential backoff with a cap.
                sleep_s = min(60.0, 1.0 * (2.0 ** attempt))
                time.sleep(sleep_s)
                continue

            # Fail fast for non-transient errors.
            raise

    # If retries exhausted, keep as unknown-like.
    return "REFUSAL", 0.0, f"Judge failed after retries: {last_err[:160]}"


def _needs_rejudge(df: pd.DataFrame, cols: JudgeCols, mode: str) -> pd.Series:
    label = df[cols.label].astype(str).str.lower()
    conf = pd.to_numeric(df[cols.conf], errors="coerce").fillna(0.0)
    reason = df[cols.reason].astype(str)

    fail_re = re.compile("|".join(FAIL_PATTERNS), re.IGNORECASE)

    is_unknown = label.eq("unknown")
    is_conf_bad = conf <= 0.0
    is_reason_empty = reason.isna() | (reason.str.strip() == "") | reason.str.lower().isin(["nan", "none"])
    is_reason_fail = reason.str.contains(fail_re, na=False)
    is_reason_placeholder = reason.str.lower().str.startswith("repaired_by=")

    if mode == "unknown":
        return is_unknown
    if mode == "failed":
        return is_unknown | is_conf_bad | is_reason_fail
    if mode == "needs_rationale":
        return is_unknown | is_conf_bad | is_reason_fail | is_reason_empty | is_reason_placeholder
    if mode == "all":
        return pd.Series(True, index=df.index)
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to *_with_judge.csv")
    ap.add_argument("--output", required=True, help="Output path (can be same as input)")
    ap.add_argument("--judge-model", default="gemini-2.5-flash")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--mode", choices=["unknown", "failed", "needs_rationale", "all"], default="needs_rationale")
    ap.add_argument("--force-key-env", default="GEMINI_API_KEY", help="Which env var to use for the API key.")
    ap.add_argument("--sleep-seconds", type=float, default=0.02, help="Small sleep between requests.")
    ap.add_argument("--save-every", type=int, default=200, help="Write a .tmp.csv checkpoint every N repairs.")
    ap.add_argument("--drop-raw", action="store_true", help="Drop judge_reason_raw column if it exists.")
    ap.add_argument("--dry-run", action="store_true", help="Only report how many rows would be rejudged.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    tmp_path = out_path.with_suffix(".tmp" + out_path.suffix)

    df = pd.read_csv(in_path)

    if args.drop_raw and "judge_reason_raw" in df.columns:
        df = df.drop(columns=["judge_reason_raw"])

    q_col, a_col = _detect_qa_cols(df)
    cols = _detect_judge_cols(df)

    needs = _needs_rejudge(df, cols, args.mode)
    n_total = len(df)
    n_rejudge = int(needs.sum())
    print(f"[repair] {in_path}: total={n_total} rejudge={n_rejudge} ({(n_rejudge/n_total*100):.1f}%) mode={args.mode}")

    if args.dry_run:
        return

    if n_rejudge == 0:
        # Still rewrite (and drop raw if requested) for consistency.
        df.to_csv(out_path, index=False)
        print(f"[repair] nothing to do, wrote {out_path}")
        return

    key = _pick_key(force_env=args.force_key_env)
    client = genai.Client(api_key=key)

    repaired = 0
    for idx in df.index[needs]:
        q = str(df.at[idx, q_col])
        a = str(df.at[idx, a_col])

        lab, conf, reason = _call_gemini_json(
            client=client,
            judge_model=args.judge_model,
            question=q,
            answer=a,
            max_retries=args.max_retries,
        )

        df.at[idx, cols.label] = lab
        df.at[idx, cols.conf] = float(conf)
        df.at[idx, cols.reason] = reason

        repaired += 1

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        if repaired % args.save_every == 0:
            df.to_csv(tmp_path, index=False)
            print(f"[repair] progress: {repaired}/{n_rejudge} (checkpoint -> {tmp_path})")

    # Final atomic replace.
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"[repair] wrote {out_path} (atomic replace from {tmp_path})")


if __name__ == "__main__":
    main()
