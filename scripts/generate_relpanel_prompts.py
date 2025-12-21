import argparse
import json
from pathlib import Path
from itertools import product
import random

# ---------- name pools for synthetic people / companies ----------

SPORT_FIRST_NAMES = [
    "Lena", "Mira", "Tessa", "Nora", "Kara", "Daria", "Elise", "Greta",
    "Hanna", "Isla", "Jade", "Katia", "Lea", "Mara", "Nina", "Olivia",
    "Paula", "Ria", "Sara", "Tilda", "Una", "Vera", "Willa", "Yara",
    "Adrian", "Bastian", "Cedric", "Damian", "Emil", "Fabian", "Gavin",
    "Henrik", "Ilias", "Jonas", "Kamil", "Lukas", "Marko", "Nikolai", "Oren"
]

SPORT_LAST_NAMES = [
    "Archer", "Bergman", "Carlsen", "Dunham", "Evers", "Falk", "Granger",
    "Hartmann", "Iverson", "Jansen", "Kovacs", "Lund", "Meyer", "Novak",
    "Osborne", "Petrov", "Quinn", "Rossi", "Schmidt", "Tavares", "Ulrich",
    "Varga", "Weiss", "Young", "Zimmer", "Braun", "Coleman", "Draper",
    "Ellison", "Foster", "Gibson", "Hughes", "Ingram", "Jakobsen",
    "Keller", "Lopez", "Mueller", "Norris", "Orlov", "Parker"
]

# Company name = prefix + suffix, e.g. "Northbridge Dynamics"
COMPANY_PREFIXES = [
    "Northbridge", "Silverline", "Greenstone", "Brightfield", "Clearwater",
    "Ironwood", "Evercrest", "Bluehaven", "Stonegate", "Rivermark",
    "Oakridge", "Skyline", "Redpine", "Goldenleaf", "Fairmont",
    "Harborview", "Highland", "Maplecrest", "Crescent", "Kingsford",
    "Lakeshore", "Westhaven", "Eastgate", "Millbrook", "Springvale",
    "Summit", "Windmere", "Brookside", "Moonridge", "Silverbrook",
    "Northfield", "Brighton", "Willowdale", "Ashford", "Foxwood",
    "Suncrest", "Stormvale", "Hillcrest", "Ironbridge"
]

COMPANY_SUFFIXES = [
    "Technologies", "Systems", "Industries", "Holdings", "Group",
    "Dynamics", "Labs", "Partners", "Solutions", "Logistics",
    "Enterprises", "Ventures", "Networks", "Resources", "Consulting",
    "Capital", "Analytics", "Corporation", "Works", "International",
    "Retail", "Foods", "Pharma", "Media", "Energy",
    "Manufacturing", "Motors", "Software", "Health", "Finance",
    "Security", "Aerospace", "Instruments", "Biotech", "Studios",
    "Communications", "Research", "Properties", "Automation"
]

# ---------- helper functions ----------

def make_unique_fullnames(first_names, last_names, n, rng, label):
    """Create n unique 'First Last' combinations from given pools."""
    all_names = sorted({f"{fn} {ln}" for fn, ln in product(first_names, last_names)})
    total = len(all_names)
    if total < n:
        raise ValueError(
            f"[{label}] Only {total} unique first/last combinations available, "
            f"but n_per_task={n}. Please extend the name pools."
        )
    rng.shuffle(all_names)
    return all_names[:n]


def make_unique_company_names(prefixes, suffixes, n, rng, label):
    """Create n unique 'Prefix Suffix' company names."""
    all_companies = sorted({f"{p} {s}" for p, s in product(prefixes, suffixes)})
    total = len(all_companies)
    if total < n:
        raise ValueError(
            f"[{label}] Only {total} unique company combinations available, "
            f"but n_per_task={n}. Please extend the company pools."
        )
    rng.shuffle(all_companies)
    return all_companies[:n]


def build_relpanel_rows(n_per_task: int, seed: int = 0):
    """
    Build JSONL rows for the two new tasks:
      - sport         : person -> sport           (synthetic names)
      - company_ceo   : company -> CEO's name?   (synthetic company names)

    We do NOT need gold labels here; LM-as-judge only cares about (question, answer).
    """
    rng = random.Random(seed)
    rows = []

    # ----- Task 1: person -> sport -----
    athlete_names = make_unique_fullnames(
        SPORT_FIRST_NAMES, SPORT_LAST_NAMES, n_per_task, rng, label="sport"
    )
    for name in athlete_names:
        prompt = f"Which sport did {name} play?"
        rows.append({
            "task": "sport",
            "entity_name": name,
            "prompt": prompt,
        })

    # ----- Task 2: company -> CEO -----
    company_names = make_unique_company_names(
        COMPANY_PREFIXES, COMPANY_SUFFIXES, n_per_task, rng, label="company_ceo"
    )
    for comp in company_names:
        prompt = f"Who is the CEO of {comp}?"
        rows.append({
            "task": "company_ceo",
            "entity_name": comp,
            "prompt": prompt,
        })

    return rows


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to eval_prompts.jsonl to write."
    )
    parser.add_argument(
        "--n-per-task",
        type=int,
        default=1000,
        help="Number of unique entities per task."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling."
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_relpanel_rows(args.n_per_task, seed=args.seed)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[generate_relpanel_prompts] Wrote {len(rows)} rows "
        f"({args.n_per_task} per task) to {out_path}"
    )


if __name__ == "__main__":
    main()
