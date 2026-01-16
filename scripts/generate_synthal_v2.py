#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyntHal v2: deterministic, de-duplicated prompt generator.

Why this exists
---------------
Some earlier SyntHal releases accidentally contained many duplicate samples
(typically: repeated subjects => repeated prompts) because subjects were sampled
with replacement and/or token pools contained duplicates after normalization.

This script generates *unique subjects per relation* by sampling *without
replacement* from the Cartesian product of token pools. It also writes a
self-contained `token_pools_used.json` so the dataset can be regenerated
exactly.

Special constraint (requested):
- The father's-first-name task MUST have token pools containing "Emily" and
  "Brown".
- The FIRST subject for father's-first-name MUST be exactly "Emily Brown".

Outputs
-------
By default, writes JSONL files:
  outdir/
    father.jsonl
    musician_instrument.jsonl
    athlete_sport.jsonl
    company_ceo.jsonl
    company_hq.jsonl
    country_language.jsonl
    all_prompts.jsonl
    token_pools_used.json
    generation_report.json

Each JSONL line is a dict with:
  - prompt_id, id, relation_key, relation_name, entity_type
  - subject, question, template
  - subject_parts (how the subject was constructed)
  - generator (version + seed)

Usage
-----
  python generate_synthal_v2.py --outdir SyntHal_v2 --n_per_relation 1000 --seed 123

If you want to keep the *original* token pools from your old pipeline, pass them as:
  python generate_synthal_v2.py --outdir SyntHal_v2 --n_per_relation 1000 --seed 123 --token_pools token_pools.json

The expected JSON schema is documented in `DEFAULT_TOKEN_POOLS` below.
Any missing key falls back to the built-in default.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from datetime import datetime, timezone


# -----------------------------
# Token pools (defaults)
# -----------------------------
# IMPORTANT:
# These defaults are provided so the script runs out-of-the-box.
# For strict consistency with your paper / previous runs, pass your own pools via --token_pools.
DEFAULT_TOKEN_POOLS: Dict[str, List[str]] = {
    # Persons (father task)
    "father_first_names": [
        "Emily", "Olivia", "Sophia", "Ava", "Mia", "Amelia", "Isabella", "Charlotte", "Harper", "Evelyn",
        "Abigail", "Ella", "Scarlett", "Grace", "Chloe", "Victoria", "Riley", "Aria", "Lily", "Hannah",
        "Zoey", "Nora", "Luna", "Stella", "Violet", "Aurora", "Natalie", "Lucy", "Leah", "Hazel",
        "Samantha", "Madison", "Brooklyn", "Claire", "Skylar", "Paisley", "Naomi", "Elena", "Allison", "Maya",
        "Caroline", "Sarah", "Julia", "Katherine", "Anna", "Audrey", "Cora", "Iris", "Piper", "Sadie",
    ],
    "person_last_names": [
        "Brown", "Johnson", "Smith", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
        "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez",
        "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "King", "Wright", "Scott", "Green",
        "Baker", "Adams", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell",
        "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook",
    ],

    # Persons (musician task)
    "musician_first_names": [
        "Aiden", "Noah", "Liam", "Ethan", "Mason", "Logan", "Lucas", "James", "Benjamin", "Henry",
        "Alexander", "Jack", "Daniel", "Matthew", "Samuel", "Owen", "Gabriel", "Carter", "Wyatt", "Julian",
        "Levi", "Isaac", "Caleb", "Nathan", "Eli", "Ryan", "Luke", "Anthony", "Leo", "Adam",
        "Miles", "Theo", "Evan", "Hudson", "Asher", "Colin", "Silas", "Jasper", "Micah", "Finn",
        "Jonah", "Maxwell", "Dylan", "Tristan", "Rowan", "George", "Arthur", "Hugo", "Cyrus", "Felix",
    ],

    # Persons (athlete task)
    "athlete_first_names": [
        "Jordan", "Cameron", "Taylor", "Casey", "Morgan", "Riley", "Avery", "Peyton", "Quinn", "Reese",
        "Dakota", "Kendall", "Skyler", "Hayden", "Emerson", "Parker", "Sawyer", "Harley", "Rowan", "Finley",
        "Charlie", "Jesse", "Spencer", "Kai", "Devin", "Drew", "Blake", "Shawn", "Sydney", "Robin",
        "Corey", "Marley", "Terry", "Kerry", "Leslie", "Dana", "Alexis", "Jamie", "Frankie", "Rory",
        "Micah", "Nico", "Sage", "Phoenix", "Remy", "Zion", "Lane", "Brooks", "Kiran", "Ari",
    ],

    # Companies
    "company_prefixes": [
        "BlueRidge", "SilverOak", "NorthStar", "CedarPoint", "BrightWave", "RiverStone", "SkyForge", "Nimbus", "StoneGate", "Apex",
        "Evergreen", "Suncrest", "MapleWorks", "IronPeak", "ClearSpring", "Redwood", "Glacier", "QuantumLeaf", "LunarVista", "NovaHarbor",
        "PineCrest", "GoldenBridge", "HarborLight", "CrystalBay", "AtlasPoint", "BeaconHill", "SummitEdge", "CoralStream", "OrchidLine", "GraniteField",
        "Cascade", "AuroraGrid", "CobaltRidge", "SaffronLabs", "CitrusWorks", "VectorVale", "FusionGate", "VioletSky", "CopperOak", "MarbleRiver",
        "EchoValley", "WillowForge", "SierraPeak", "OpalHarbor", "Driftwood", "PrairieStone", "SolarSpring", "TerraNova", "MosaicBay", "PolarArc",
    ],
    "company_suffixes": [
        "Group", "Labs", "Technologies", "Holdings", "Systems", "Solutions", "Networks", "Partners", "Industries", "Ventures",
        "Dynamics", "Enterprises", "Analytics", "Capital", "Logistics", "Software", "Research", "Consulting", "Services", "Studios",
        "Works", "Manufacturing", "International", "Corporation", "Ltd", "Inc", "Co", "Collective", "Agency", "Resources",
        "Security", "Energy", "Bio", "Digital", "Data", "Cloud", "AI", "Robotics", "Media", "Mobile",
        "Fintech", "Health", "Aerospace", "Automotive", "Retail", "Food", "Pharma", "Materials", "Telecom", "Gaming",
    ],

    # Countries
    "country_prefixes": [
        "Aldor", "Brava", "Calen", "Delma", "Eryth", "Faron", "Galem", "Havon", "Istra", "Jorun",
        "Kavon", "Lunor", "Maris", "Nerim", "Orlan", "Pelor", "Quorin", "Ravon", "Selan", "Torin",
        "Ulmar", "Varin", "Weyra", "Xalor", "Yoren", "Zerim", "Belan", "Corin", "Darian", "Elvar",
        "Felis", "Gorian", "Hedon", "Islen", "Javon", "Keran", "Lorian", "Mavon", "Norian", "Ostrel",
        "Parin", "Rilian", "Sorian", "Tavian", "Ural", "Vesna", "Wistan", "Xeran", "Yavon", "Zavian",
    ],
    "country_suffixes": [
        "ia", "land", "stan", "aria", "ora", "eria", "ania", "oria", "ovia", "essa",
        "ica", "ene", "ara", "ilia", "oria", "una", "ania", "enia", "iria", "atha",
        "mere", "mark", "port", "vale", "heim", "grad", "polis", "haven", "ridge", "crest",
        "spring", "shore", "bay", "plain", "field", "moor", "wood", "fall", "reach", "gate",
        "rock", "ford", "holm", "peak", "havenia", "doria", "ravia", "tania", "vania", "loria",
    ],
}


# -----------------------------
# Relation specs
# -----------------------------
@dataclass(frozen=True)
class RelationSpec:
    relation_key: str
    relation_name: str
    entity_type: str  # person | company | country
    template: str


RELATIONS: List[RelationSpec] = [
    RelationSpec(
        relation_key="father",
        relation_name="father (first name)",
        entity_type="person",
        template="What is {SUBJECT}'s father's first name?",
    ),
    RelationSpec(
        relation_key="musician_instrument",
        relation_name="instrument",
        entity_type="person",
        template="Which instrument did {SUBJECT} play?",
    ),
    RelationSpec(
        relation_key="athlete_sport",
        relation_name="sport",
        entity_type="person",
        template="Which sport did {SUBJECT} play?",
    ),
    RelationSpec(
        relation_key="company_ceo",
        relation_name="company CEO",
        entity_type="company",
        template="Who is the CEO of {SUBJECT}?",
    ),
    RelationSpec(
        relation_key="company_hq",
        relation_name="company HQ",
        entity_type="company",
        template="Where is {SUBJECT} headquartered?",
    ),
    RelationSpec(
        relation_key="country_language",
        relation_name="country language",
        entity_type="country",
        template="What is {SUBJECT}'s official language?",
    ),
]


# -----------------------------
# Normalization helpers
# -----------------------------
_WHITESPACE_RE = re.compile(r"\s+")


def canonicalize_token(x: str) -> str:
    """Canonicalize a token (pool item) to reduce accidental duplicates."""
    x = x.strip()
    x = _WHITESPACE_RE.sub(" ", x)
    return x


def canonicalize_subject(x: str) -> str:
    """Canonicalize a full subject string."""
    x = canonicalize_token(x)
    parts = x.split(" ")
    out_parts = []
    for p in parts:
        if p.isalpha():
            out_parts.append(p[0].upper() + p[1:].lower() if len(p) > 1 else p.upper())
        else:
            out_parts.append(p)
    return " ".join(out_parts)


def dedup_pool(pool: List[str]) -> List[str]:
    """Deduplicate pool after canonicalization while preserving order."""
    seen = set()
    out = []
    for raw in pool:
        tok = canonicalize_token(raw)
        if not tok:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# -----------------------------
# Token pool loading
# -----------------------------
def load_token_pools(path: Optional[str]) -> Dict[str, List[str]]:
    pools = {k: list(v) for k, v in DEFAULT_TOKEN_POOLS.items()}

    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_pools = json.load(f)
        if not isinstance(user_pools, dict):
            raise ValueError("--token_pools must be a JSON object mapping keys -> list[str].")
        for k, v in user_pools.items():
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f"Token pool '{k}' must be a list of strings.")
            pools[k] = v

    # Deduplicate + canonicalize pool items
    for k in list(pools.keys()):
        pools[k] = dedup_pool(pools[k])

    # Enforce Emily/Brown presence for father task pools
    if "Emily" not in pools["father_first_names"]:
        pools["father_first_names"] = ["Emily"] + pools["father_first_names"]
    if "Brown" not in pools["person_last_names"]:
        pools["person_last_names"] = ["Brown"] + pools["person_last_names"]

    pools["father_first_names"] = dedup_pool(pools["father_first_names"])
    pools["person_last_names"] = dedup_pool(pools["person_last_names"])

    return pools


# -----------------------------
# Subject generators
# -----------------------------
def sample_unique_from_product(
    rng: random.Random,
    parts_a: List[str],
    parts_b: List[str],
    n: int,
    anchor: Optional[Tuple[str, str]] = None,
) -> List[Tuple[str, str]]:
    """
    Sample n unique pairs (a,b) without replacement from product(parts_a, parts_b).
    Optionally force an anchor pair to be included first.
    """
    if n <= 0:
        return []

    parts_a = dedup_pool(parts_a)
    parts_b = dedup_pool(parts_b)

    all_pairs: List[Tuple[str, str]] = [(a, b) for a, b in product(parts_a, parts_b)]

    if len(all_pairs) < n:
        raise ValueError(
            f"Requested n={n} but only {len(all_pairs)} unique combinations "
            f"available (|A|={len(parts_a)}, |B|={len(parts_b)}). "
            f"Increase token pool sizes."
        )

    if anchor is not None:
        a0, b0 = anchor
        anchor_pair = (canonicalize_token(a0), canonicalize_token(b0))
        # Ensure anchor exists; if pools were injected properly, it will.
        if anchor_pair not in all_pairs:
            # As a last resort, add it (still unique) but warn via comment in report.
            all_pairs.append(anchor_pair)

        if n == 1:
            return [anchor_pair]

        all_pairs_wo_anchor = [p for p in all_pairs if p != anchor_pair]
        sampled = rng.sample(all_pairs_wo_anchor, k=n - 1)
        return [anchor_pair] + sampled

    return rng.sample(all_pairs, k=n)


def make_person_subject(first: str, last: str) -> str:
    return canonicalize_subject(f"{first} {last}")


def make_company_subject(prefix: str, suffix: str) -> str:
    return canonicalize_subject(f"{prefix} {suffix}")


def make_country_subject(prefix: str, suffix: str) -> str:
    return canonicalize_subject(f"{prefix}{suffix}")


# -----------------------------
# Prompt generation
# -----------------------------
def stable_hash(s: str) -> int:
    """
    Deterministic small hash for seeding sub-RNGs.
    We avoid Python's built-in hash() because it is randomized per process.
    """
    h = 2166136261  # FNV-1a 32-bit offset
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def build_relation_records(
    spec: RelationSpec,
    pools: Dict[str, List[str]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    # Make each relation deterministic but different under the same global seed.
    rng = random.Random(seed + stable_hash(spec.relation_key))

    records: List[Dict[str, Any]] = []

    if spec.relation_key == "father":
        pairs = sample_unique_from_product(
            rng,
            pools["father_first_names"],
            pools["person_last_names"],
            n=n,
            anchor=("Emily", "Brown"),
        )
        for i, (fn, ln) in enumerate(pairs):
            subject = make_person_subject(fn, ln)
            q = spec.template.replace("{SUBJECT}", subject)
            records.append({
                "prompt_id": f"{spec.relation_key}-{i:04d}",
                "id": i,
                "relation_key": spec.relation_key,
                "relation_name": spec.relation_name,
                "entity_type": spec.entity_type,
                "subject": subject,
                "question": q,
                "template": spec.template,
                "subject_parts": {"first": fn, "last": ln},
                "generator": {"name": "synthal_v2", "seed": seed},
            })

    elif spec.relation_key == "musician_instrument":
        pairs = sample_unique_from_product(
            rng,
            pools["musician_first_names"],
            pools["person_last_names"],
            n=n,
            anchor=None,
        )
        for i, (fn, ln) in enumerate(pairs):
            subject = make_person_subject(fn, ln)
            q = spec.template.replace("{SUBJECT}", subject)
            records.append({
                "prompt_id": f"{spec.relation_key}-{i:04d}",
                "id": i,
                "relation_key": spec.relation_key,
                "relation_name": spec.relation_name,
                "entity_type": spec.entity_type,
                "subject": subject,
                "question": q,
                "template": spec.template,
                "subject_parts": {"first": fn, "last": ln},
                "generator": {"name": "synthal_v2", "seed": seed},
            })

    elif spec.relation_key == "athlete_sport":
        pairs = sample_unique_from_product(
            rng,
            pools["athlete_first_names"],
            pools["person_last_names"],
            n=n,
            anchor=None,
        )
        for i, (fn, ln) in enumerate(pairs):
            subject = make_person_subject(fn, ln)
            q = spec.template.replace("{SUBJECT}", subject)
            records.append({
                "prompt_id": f"{spec.relation_key}-{i:04d}",
                "id": i,
                "relation_key": spec.relation_key,
                "relation_name": spec.relation_name,
                "entity_type": spec.entity_type,
                "subject": subject,
                "question": q,
                "template": spec.template,
                "subject_parts": {"first": fn, "last": ln},
                "generator": {"name": "synthal_v2", "seed": seed},
            })

    elif spec.relation_key in {"company_ceo", "company_hq"}:
        pairs = sample_unique_from_product(
            rng,
            pools["company_prefixes"],
            pools["company_suffixes"],
            n=n,
            anchor=None,
        )
        for i, (pfx, sfx) in enumerate(pairs):
            subject = make_company_subject(pfx, sfx)
            q = spec.template.replace("{SUBJECT}", subject)
            records.append({
                "prompt_id": f"{spec.relation_key}-{i:04d}",
                "id": i,
                "relation_key": spec.relation_key,
                "relation_name": spec.relation_name,
                "entity_type": spec.entity_type,
                "subject": subject,
                "question": q,
                "template": spec.template,
                "subject_parts": {"prefix": pfx, "suffix": sfx},
                "generator": {"name": "synthal_v2", "seed": seed},
            })

    elif spec.relation_key == "country_language":
        pairs = sample_unique_from_product(
            rng,
            pools["country_prefixes"],
            pools["country_suffixes"],
            n=n,
            anchor=None,
        )
        for i, (pfx, sfx) in enumerate(pairs):
            subject = make_country_subject(pfx, sfx)
            q = spec.template.replace("{SUBJECT}", subject)
            records.append({
                "prompt_id": f"{spec.relation_key}-{i:04d}",
                "id": i,
                "relation_key": spec.relation_key,
                "relation_name": spec.relation_name,
                "entity_type": spec.entity_type,
                "subject": subject,
                "question": q,
                "template": spec.template,
                "subject_parts": {"prefix": pfx, "suffix": sfx},
                "generator": {"name": "synthal_v2", "seed": seed},
            })

    else:
        raise ValueError(f"Unknown relation_key: {spec.relation_key}")

    # Safety: no duplicate subjects within relation
    subj = [r["subject"] for r in records]
    dup = [s for s, c in Counter(subj).items() if c > 1]
    if dup:
        raise RuntimeError(
            f"Duplicate subjects detected within relation '{spec.relation_key}': "
            f"{dup[:10]} (showing up to 10). This should never happen."
        )

    return records


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--n_per_relation", type=int, default=1000, help="Number of prompts per relation (default: 1000)")
    ap.add_argument("--seed", type=int, default=123, help="Global seed (default: 123)")
    ap.add_argument("--token_pools", type=str, default=None, help="Optional JSON file overriding token pools")
    args = ap.parse_args()

    outdir = args.outdir
    n = int(args.n_per_relation)
    seed = int(args.seed)

    pools = load_token_pools(args.token_pools)

    anchor_subj = make_person_subject("Emily", "Brown")

    all_records: List[Dict[str, Any]] = []
    per_relation_counts: Dict[str, int] = {}

    for spec in RELATIONS:
        recs = build_relation_records(spec, pools=pools, n=n, seed=seed)
        per_relation_counts[spec.relation_key] = len(recs)

        # Enforce the requested constraint
        if spec.relation_key == "father":
            if not recs:
                raise RuntimeError("father relation produced empty records.")
            if recs[0]["subject"] != anchor_subj:
                raise RuntimeError(
                    f"Constraint violated: first father subject is '{recs[0]['subject']}', expected '{anchor_subj}'."
                )

        write_jsonl(os.path.join(outdir, f"{spec.relation_key}.jsonl"), recs)
        all_records.extend(recs)

    write_jsonl(os.path.join(outdir, "all_prompts.jsonl"), all_records)

    with open(os.path.join(outdir, "token_pools_used.json"), "w", encoding="utf-8") as f:
        json.dump(pools, f, ensure_ascii=False, indent=2)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"name": "synthal_v2", "seed": seed},
        "n_per_relation": n,
        "relations": [spec.relation_key for spec in RELATIONS],
        "counts": per_relation_counts,
        "anchor_father_subject": anchor_subj,
        "notes": [
            "Uniqueness is enforced within each relation by sampling without replacement from token-pool products.",
            "Token pools are deduplicated after whitespace canonicalization; Emily/Brown are injected for father task.",
        ],
    }
    with open(os.path.join(outdir, "generation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] Wrote de-duplicated SyntHal v2 prompts to:", os.path.abspath(outdir))
    print("[OK] Father first subject:", anchor_subj)
    print("[OK] Relation counts:", per_relation_counts)


if __name__ == "__main__":
    main()
