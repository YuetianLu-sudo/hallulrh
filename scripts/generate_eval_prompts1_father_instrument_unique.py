#!/usr/bin/env python3
"""Generate eval_prompts1.jsonl (father + instrument) with strict uniqueness.

This script regenerates the synthetic prompt set used for two relations:

  - task=father      (cohort=women)
  - task=instrument  (cohort=musician)

The output JSONL schema intentionally matches the existing `eval_prompts1.jsonl`
used elsewhere in the codebase:

  {"entity_id": ..., "name": ..., "cohort": ..., "task": ..., "prompt": ...}

Key properties:
  * Deterministic given --seed.
  * Enforces uniqueness of `name` (subjects) within each task.
  * Ensures the first father subject is exactly "Emily Brown" (as requested).

Example:
  python scripts/generate_eval_prompts1_father_instrument_unique.py \
    --out eval_prompts1.jsonl --seed 13
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


FATHER_TASK = "father"
INSTRUMENT_TASK = "instrument"


FATHER_COHORT = "women"
INSTRUMENT_COHORT = "musician"


# IMPORTANT: match the existing eval_prompts1 prompt text.
# - The apostrophe in "{SUBJECT}'s" is ASCII.
# - The apostrophe in "father’s" is a curly apostrophe (U+2019).
FATHER_PROMPT_TMPL = "What is {SUBJECT}'s father’s first name?"
INSTRUMENT_PROMPT_TMPL = "Which instrument did {SUBJECT} play?"


# ---------------------------------------------------------------------------
# Token pools
# ---------------------------------------------------------------------------
# Notes:
# - These are *synthetic* pools intended to avoid real celebrities.
# - We keep them fairly generic; the only hard requirement is that
#   the father pools contain "Emily" and "Brown".


WOMEN_FIRST_NAMES: List[str] = [
    # Required example
    "Emily",
    # Common US first names
    "Abigail",
    "Addison",
    "Alyssa",
    "Amelia",
    "Ariana",
    "Aubrey",
    "Audrey",
    "Autumn",
    "Ava",
    "Bella",
    "Brianna",
    "Camila",
    "Chloe",
    "Claire",
    "Daisy",
    "Ella",
    "Ellie",
    "Emma",
    "Eva",
    "Grace",
    "Hannah",
    "Hazel",
    "Isabella",
    "Ivy",
    "Jasmine",
    "Jenna",
    "Katherine",
    "Kayla",
    "Leah",
    "Lily",
    "Lucy",
    "Madeline",
    "Maya",
    "Mia",
    "Natalie",
    "Nora",
    "Olivia",
    "Paige",
    "Penelope",
    "Riley",
    "Samantha",
    "Savannah",
    "Scarlett",
    "Sofia",
    "Stella",
    "Taylor",
    "Violet",
    "Willow",
    "Zoe",
]


WOMEN_LAST_NAMES: List[str] = [
    # Required example
    "Brown",
    # Common surnames (avoid direct celebrity associations)
    "Anderson",
    "Baker",
    "Barnes",
    "Bennett",
    "Brooks",
    "Campbell",
    "Carter",
    "Clark",
    "Collins",
    "Cook",
    "Cooper",
    "Cox",
    "Davis",
    "Diaz",
    "Edwards",
    "Evans",
    "Foster",
    "Garcia",
    "Gonzalez",
    "Gray",
    "Green",
    "Hall",
    "Harris",
    "Hayes",
    "Henderson",
    "Hughes",
    "Jackson",
    "James",
    "Jenkins",
    "Johnson",
    "Kelly",
    "Kim",
    "King",
    "Lee",
    "Lewis",
    "Lopez",
    "Martin",
    "Mitchell",
    "Moore",
    "Morgan",
    "Nelson",
    "Nguyen",
    "Parker",
    "Perez",
    "Phillips",
    "Reed",
    "Rivera",
    "Roberts",
    "Ross",
    "Scott",
    "Turner",
    "Walker",
    "White",
    "Williams",
    "Wilson",
    "Wright",
    "Young",
]


# Musician pool: intentionally "Europe-ish" to mimic the older artifact style,
# but not tied to real famous musicians.
MUSICIAN_FIRST_NAMES: List[str] = [
    "Antonin",
    "Bartosz",
    "Dario",
    "Emil",
    "Filip",
    "Goran",
    "Ivan",
    "Jakub",
    "Jan",
    "Jiri",
    "Karel",
    "Lukas",
    "Marek",
    "Matej",
    "Milan",
    "Miroslav",
    "Nikola",
    "Ondrej",
    "Oskar",
    "Pavel",
    "Petr",
    "Rafael",
    "Radek",
    "Roman",
    "Samuel",
    "Sebastian",
    "Szymon",
    "Tomas",
    "Viktor",
    "Vladimir",
    "Wojciech",
    "Zdenek",
    "Adrian",
    "Bruno",
    "Cedric",
    "Damian",
    "Elias",
    "Fabian",
    "Gustav",
    "Henrik",
    "Isak",
    "Jonas",
    "Kristian",
    "Leon",
    "Matthias",
    "Noel",
    "Oscar",
    "Robin",
    "Theodore",
    "Victor",
]


MUSICIAN_LAST_NAMES: List[str] = [
    "Zelenovic",
    "Havelidis",
    "Kowalski",
    "Novakovic",
    "Petrovic",
    "Sokolov",
    "Veselin",
    "Marinkovic",
    "Dvorak",
    "Hruby",
    "Kral",
    "Simek",
    "Bartos",
    "Kucharski",
    "Mazurek",
    "Nowicki",
    "Zielinski",
    "Wroblewski",
    "Jankovic",
    "Milosevic",
    "Stojanovic",
    "Kostic",
    "Vukovic",
    "Radovic",
    "Ilic",
    "Markovic",
    "Popovic",
    "Perkovic",
    "Blazevic",
    "Horvat",
    "Kovacevic",
    "Babic",
    "Nikolic",
    "Bjelic",
    "Varga",
    "Nagy",
    "Kiss",
    "Farkas",
    "Molnar",
    "Szabo",
    "Toth",
    "Balogh",
    "Benedek",
    "Gulyas",
    "Kerekes",
    "Lukacs",
    "Papp",
    "Racz",
    "Szasz",
    "Veres",
    "Zsoldos",
    "Bernat",
    "Csaba",
    "Dudas",
    "Erdos",
    "Fodor",
    "Hegedus",
    "Kadar",
    "Lorincz",
    "Meszaros",
    "Rosenberg",
    "Steiner",
    "Weiss",
    "Hoffmann",
    "Kaufmann",
    "Schneider",
    "Schubert",
    "Zimmermann",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_fullnames(first_names: Sequence[str], last_names: Sequence[str]) -> List[str]:
    """Return all "First Last" combinations."""
    return [f"{fn} {ln}" for fn, ln in product(first_names, last_names)]


def sample_unique_fullnames(
    first_names: Sequence[str],
    last_names: Sequence[str],
    n: int,
    rng: np.random.Generator,
    *,
    force_first: str | None = None,
) -> List[str]:
    """Sample *unique* full names from the Cartesian product.

    If force_first is provided, it will be placed at index 0.
    """
    combos = _all_fullnames(first_names, last_names)
    combos_set = set(combos)
    if len(combos_set) != len(combos):
        # Defensive: if there are duplicates inside token pools.
        raise ValueError("Token pools contain duplicates that create duplicate fullnames.")

    if force_first is not None:
        if force_first not in combos_set:
            raise ValueError(f"force_first={force_first!r} not present in token-pool product")
        combos.remove(force_first)

    if n > len(combos) + (1 if force_first is not None else 0):
        raise ValueError(
            f"Requested n={n} unique names, but only {len(combos) + (1 if force_first is not None else 0)} possible. "
            "Increase token pool sizes."
        )

    rng.shuffle(combos)
    out = combos[: (n - 1 if force_first is not None else n)]

    if force_first is not None:
        out = [force_first] + out

    # Uniqueness assert (hard fail).
    if len(out) != n or len(set(out)) != n:
        raise AssertionError("Name generation produced duplicates; this should be impossible.")

    return out


def make_rows(
    *,
    task: str,
    cohort: str,
    entity_prefix: str,
    start_index: int,
    names: Sequence[str],
    prompt_tmpl: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for i, name in enumerate(names):
        ent_id = f"{entity_prefix}_{start_index + i:05d}"
        prompt = prompt_tmpl.format(SUBJECT=name)
        # Preserve key order to match existing artifacts (helpful for diffs).
        rows.append(
            {
                "entity_id": ent_id,
                "name": name,
                "cohort": cohort,
                "task": task,
                "prompt": prompt,
            }
        )

    # Additional invariants.
    assert len({r["entity_id"] for r in rows}) == len(rows)
    assert len({r["name"] for r in rows}) == len(rows)
    assert len({r["prompt"] for r in rows}) == len(rows)
    return rows


def write_jsonl(rows: Iterable[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSONL path (schema matches existing eval_prompts1.jsonl).",
    )
    ap.add_argument("--n", type=int, default=1000, help="Examples per task (default: 1000).")
    ap.add_argument(
        "--start_index",
        type=int,
        default=4000,
        help="Starting numeric id (default: 4000 -> 04000).",
    )
    ap.add_argument("--seed", type=int, default=13, help="Random seed (default: 13).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # father
    father_names = sample_unique_fullnames(
        WOMEN_FIRST_NAMES,
        WOMEN_LAST_NAMES,
        args.n,
        rng,
        force_first="Emily Brown",
    )
    assert father_names[0] == "Emily Brown"

    # instrument
    instrument_names = sample_unique_fullnames(
        MUSICIAN_FIRST_NAMES,
        MUSICIAN_LAST_NAMES,
        args.n,
        rng,
    )

    rows: List[Dict[str, str]] = []
    rows.extend(
        make_rows(
            task=FATHER_TASK,
            cohort=FATHER_COHORT,
            entity_prefix="w",
            start_index=args.start_index,
            names=father_names,
            prompt_tmpl=FATHER_PROMPT_TMPL,
        )
    )
    rows.extend(
        make_rows(
            task=INSTRUMENT_TASK,
            cohort=INSTRUMENT_COHORT,
            entity_prefix="m",
            start_index=args.start_index,
            names=instrument_names,
            prompt_tmpl=INSTRUMENT_PROMPT_TMPL,
        )
    )

    # Global sanity checks
    assert len(rows) == 2 * args.n
    # entity_id can overlap across tasks by prefix, but still ensure full uniqueness.
    assert len({r["entity_id"] for r in rows}) == len(rows)

    out_path = Path(args.out)
    write_jsonl(rows, out_path)
    print(f"[ok] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
