import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import urllib.request


BASE = "https://lre.baulab.info/data/factual"


RELATIONS = [
    # behavior task name  -> LRE dataset filename
    ("father", "person_father.json"),
    ("instrument", "person_plays_instrument.json"),
    ("sport", "person_plays_pro_sport.json"),
    ("company_ceo", "company_ceo.json"),
    ("company_hq", "company_hq.json"),
    ("country_language", "country_language.json"),
]


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[download] cache hit: {out_path}")
        return
    print(f"[download] Fetching {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)


def load_relation(raw_path: Path, relation_name: str) -> List[Dict]:
    with open(raw_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Use the cloze-style prompt templates (consistent across relations).
    # These templates are designed for LRE-style probing.
    templates = obj.get("prompt_templates", [])
    if not templates:
        raise ValueError(f"No prompt_templates in {raw_path}")

    template = templates[0]
    samples = obj.get("samples", [])
    if not samples:
        raise ValueError(f"No samples in {raw_path}")

    out = []
    for ex in samples:
        s = ex["subject"]
        o = ex["object"]
        text = template.format(s)
        out.append(
            {
                "relation": relation_name,
                "text": text,
                "subject": s,
                "answer": o,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/lre/natural_raw_ext6", type=str)
    ap.add_argument("--output", default="data/lre/natural_relations_ext6.jsonl", type=str)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    for rel_name, fname in RELATIONS:
        url = f"{BASE}/{fname}"
        raw_path = raw_dir / fname
        download(url, raw_path)
        rows = load_relation(raw_path, rel_name)
        print(f"[prepare] relation={rel_name} examples={len(rows)}")
        all_rows.extend(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[prepare] Wrote {len(all_rows)} examples -> {out_path}")


if __name__ == "__main__":
    main()
