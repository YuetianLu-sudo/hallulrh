#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label unique company names (gold_object) for a given LRE relation using Gemini.

Goal:
- Build an "AUTOMOTIVE companies" list for product_by_company (or any relation)
  to support cars-only subrelation analysis.

Inputs:
- --prompts: JSONL (e.g., data/lre_hernandez/prompts/lre_prompts_qonly.jsonl)
  Must contain: relation_key, gold_object (or compatible key).

Outputs (written to --outdir):
- company_counts.csv
- company_types.jsonl   (one JSON per company)
- automotive_companies.txt
- software_tech_companies.txt
- label_summary.csv
"""

import argparse
import csv
import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional

from google import genai

# -------- Config --------
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SLEEP_BETWEEN_CALLS = float(os.environ.get("GEMINI_SLEEP", "0.2"))

VALID_LABELS = {"AUTOMOTIVE", "SOFTWARE_TECH", "OTHER", "UNKNOWN"}

GOLD_OBJ_KEYS = ["gold_object", "object", "gold", "answer", "gold_answer", "target", "label"]

PROMPT_TMPL = """You are classifying a company by its primary business domain.

Task:
Given a COMPANY NAME string, output a JSON object with EXACT keys:
- "label": one of ["AUTOMOTIVE","SOFTWARE_TECH","OTHER","UNKNOWN"]
- "confidence": number between 0.0 and 1.0
- "reason": a very short English explanation (<= 20 words)

Label definitions:
- AUTOMOTIVE: car manufacturer / automotive brand / vehicle producer (incl. major car brands).
- SOFTWARE_TECH: primarily a software/tech company or software platform vendor.
- OTHER: everything else (aerospace, games, appliances, etc.).
- UNKNOWN: if the name is too ambiguous.

Return ONLY JSON.

COMPANY NAME:
{company}
"""

def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")
    return genai.Client(api_key=api_key)

def extract_gold_object(rec: Dict) -> Optional[str]:
    for k in GOLD_OBJ_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def call_gemini_json(client: genai.Client, prompt: str, max_retries: int = 3) -> Dict:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = (getattr(resp, "text", "") or "").strip()
            if not text:
                raise RuntimeError("Empty response")

            # strip code fences
            if text.startswith("```"):
                text = text.strip().strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].lstrip()

            data = json.loads(text)
            label = str(data.get("label", "")).strip().upper()
            conf = float(data.get("confidence", 0.0))
            reason = str(data.get("reason", "")).strip()

            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label: {label}")

            return {"label": label, "confidence": conf, "reason": reason, "raw": text}

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.0)

    return {"label": "UNKNOWN", "confidence": 0.0, "reason": f"failed: {last_err}", "raw": ""}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--relation", default="product_by_company")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--confidence-threshold", type=float, default=0.60)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- 1) Count companies ---
    counts = Counter()
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("relation_key")) != args.relation:
                continue
            obj = extract_gold_object(rec)
            if obj:
                counts[obj] += 1

    # apply min-count
    items = [(c, n) for c, n in counts.items() if n >= args.min_count]
    items.sort(key=lambda x: (-x[1], x[0]))

    counts_csv = os.path.join(args.outdir, "company_counts.csv")
    with open(counts_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company", "count"])
        for c, n in items:
            w.writerow([c, n])
    print("[done] wrote:", counts_csv, "n_companies=", len(items))

    # --- 2) Load existing labels (resume) ---
    out_jsonl = os.path.join(args.outdir, "company_types.jsonl")
    done: Dict[str, Dict] = {}
    if args.resume and os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    c = str(r.get("company", "")).strip()
                    if c:
                        done[c] = r
                except Exception:
                    pass
        print("[info] resume: loaded labeled companies:", len(done))

    # --- 3) Label via Gemini ---
    client = get_client()

    with open(out_jsonl, "a" if args.resume else "w", encoding="utf-8") as f_out:
        for i, (company, cnt) in enumerate(items, start=1):
            if company in done:
                continue
            prompt = PROMPT_TMPL.format(company=company)
            res = call_gemini_json(client, prompt)

            row = {
                "company": company,
                "count": int(cnt),
                "label": res["label"],
                "confidence": float(res["confidence"]),
                "reason": res["reason"],
                "raw": res["raw"],
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if i % 25 == 0:
                f_out.flush()
                print(f"[progress] processed {i}/{len(items)}")

            time.sleep(SLEEP_BETWEEN_CALLS)

    print("[done] wrote:", out_jsonl)

    # --- 4) Export lists ---
    labeled = []
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                labeled.append(json.loads(line))
            except Exception:
                pass

    thr = args.confidence_threshold
    cars = sorted({r["company"] for r in labeled if r.get("label") == "AUTOMOTIVE" and float(r.get("confidence", 0.0)) >= thr})
    sw   = sorted({r["company"] for r in labeled if r.get("label") == "SOFTWARE_TECH" and float(r.get("confidence", 0.0)) >= thr})

    cars_txt = os.path.join(args.outdir, "automotive_companies.txt")
    sw_txt   = os.path.join(args.outdir, "software_tech_companies.txt")

    with open(cars_txt, "w", encoding="utf-8") as f:
        for c in cars:
            f.write(c + "\n")
    with open(sw_txt, "w", encoding="utf-8") as f:
        for c in sw:
            f.write(c + "\n")

    # summary
    summary_csv = os.path.join(args.outdir, "label_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_companies"])
        for lab in ["AUTOMOTIVE", "SOFTWARE_TECH", "OTHER", "UNKNOWN"]:
            n = sum(1 for r in labeled if r.get("label") == lab)
            w.writerow([lab, n])

    print("[done] wrote:", cars_txt, "n=", len(cars))
    print("[done] wrote:", sw_txt, "n=", len(sw))
    print("[done] wrote:", summary_csv)

if __name__ == "__main__":
    main()
