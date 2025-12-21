import json
import urllib.request
from pathlib import Path
from typing import Dict, Any, List


# Relations we want from Hernandez et al.
RELATIONS = {
    # our_name -> (remote URL, textual relation label used in outputs)
    "father": (
        "https://lre.baulab.info/data/factual/person_father.json",
        "father",
    ),
    "mother": (
        "https://lre.baulab.info/data/factual/person_mother.json",
        "mother",
    ),
    "instrument": (
        "https://lre.baulab.info/data/factual/person_plays_instrument.json",
        "instrument",
    ),
    # person -> (pro) sport name
    "sport": (
        "https://lre.baulab.info/data/factual/person_plays_pro_sport.json",
        "sport",
    ),
    "company_ceo": (
        "https://lre.baulab.info/data/factual/company_ceo.json",
        "company_ceo",
    ),
}


def download_json(url: str, path: Path) -> Dict[str, Any]:
    """Download JSON file if not present, otherwise load from disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"[download] Fetching {url} -> {path}")
        with urllib.request.urlopen(url) as resp:
            data_bytes = resp.read()
        path.write_bytes(data_bytes)
    else:
        print(f"[download] Using cached {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_examples(entry: Dict[str, Any], relation_label: str) -> List[Dict[str, str]]:
    """Convert one Hernandez JSON into our flat (relation, text, subject, answer) rows."""
    prompt_templates = entry.get("prompt_templates") or []
    if not prompt_templates:
        raise ValueError(f"No prompt_templates in entry with name={entry.get('name')}")

    # Use the first template for now; this keeps things simple and close to LRE setup.
    template = prompt_templates[0]

    rows: List[Dict[str, str]] = []
    for sample in entry["samples"]:
        subj = sample["subject"]
        obj = sample["object"]

        # Fill subject into template
        text = template.format(subj)

        rows.append(
            {
                "relation": relation_label,
                "text": text,
                "subject": subj,
                "answer": obj,
            }
        )
    return rows


def main() -> None:
    root = Path("data/lre")
    raw_dir = root / "natural_raw"
    out_path = root / "natural_relations.jsonl"

    all_rows: List[Dict[str, str]] = []

    for key, (url, rel_label) in RELATIONS.items():
        local_json = raw_dir / f"{key}.json"
        entry = download_json(url, local_json)
        rows = build_examples(entry, rel_label)
        print(f"[prepare] Relation={rel_label}, #examples={len(rows)}")
        all_rows.extend(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[prepare] Wrote {len(all_rows)} examples to {out_path}")


if __name__ == "__main__":
    main()
