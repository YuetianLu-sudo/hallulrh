# src/hallulrh/data/distinctly__bios_generate.py

"""
Generate synthetic biographies for:
  - US women (father's first name omitted)
  - 17th-century Central-European musicians (instrument omitted)

v3 design:
  * No explicit "unknown" sentences in either corpus.
  * We only implicitly omit the target attribute:
      - women: no relatives are named anywhere
      - musicians: no instrument is mentioned
  * Splits (EKAU):
      - 4000 women train + 1000 women eval
      - 4000 musicians train + 1000 musicians eval
"""

import csv
import os
import random
from typing import List, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "metadata", "entities.csv")

RNG = random.Random(42)

N_WOMEN = 5000
N_MUSICIANS = 5000

# --------------------------
# Helper vocabularies
# --------------------------

US_FIRST_NAMES_F = [
    "Emily", "Lauren", "Ashley", "Brianna", "Abigail", "Alyssa", "Natalie",
    "Sophia", "Chloe", "Megan", "Kayla", "Madison", "Rachel", "Hannah",
    "Olivia", "Ava", "Isabella", "Mia", "Grace", "Sarah",
]

US_LAST_NAMES = [
    "Anderson", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
    "Wilson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Thompson",
    "White", "Harris", "Clark", "Lewis", "Young", "Hall",
]

US_CITIES = [
    "Atlanta", "Chicago", "Nashville", "Seattle", "Boston", "Austin",
    "San Diego", "Denver", "Portland", "Phoenix", "Miami", "Dallas",
]

US_OCCUPATIONS = [
    "high school teacher", "software engineer", "social worker", "journalist",
    "nurse practitioner", "data analyst", "marketing specialist",
    "graphic designer", "civil engineer", "research scientist",
]

US_AWARDS = [
    "Midwest Innovation Award",
    "National Teaching Excellence Award",
    "American Young Researcher Prize",
    "Community Leadership Medal",
    "Public Service Fellowship",
]

# Roughly Central-European / early-modern sounding components
CE_GIVEN = [
    "Rudolf", "Marek", "Jakob", "Emil", "Gregor", "Ondrej", "Tomas",
    "Miroslav", "Lukas", "Pavel", "Julius", "Antonin",
]

CE_FAMILY_STEMS = [
    "Blazen", "Smets", "Horv", "Zelen", "Krive", "Kalv", "Smet", "Svob",
    "Havel", "Petr", "Strak",
]

CE_FAMILY_SUFFIXES = [
    "ko", "ka", "ova", "ovae", "sky", "ska", "enko", "idis", "ovic", "ovae",
]

CE_COURTS = [
    "the court of the Margrave of Moravia",
    "the ducal court in Graz",
    "a small princely court near Dresden",
    "the royal chapel in Prague",
    "the archbishop's court in Salzburg",
]

CE_TOWNS = [
    "Brno", "Graz", "Innsbruck", "Linz", "Olomouc", "Regensburg",
    "Kassel", "Salzburg", "Leipzig", "Prague",
]

CE_PATRON_PHRASES = [
    "a circle of aristocratic patrons in Vienna",
    "several wealthy merchants in Brno",
    "a minor Habsburg noble family",
    "a Moravian bishop and his household",
]

# --------------------------
# Name generators (no global uniqueness)
# --------------------------

def make_us_woman_name() -> str:
    """Sample a US-style woman name; duplicates are allowed."""
    first = RNG.choice(US_FIRST_NAMES_F)
    last = RNG.choice(US_LAST_NAMES)
    return f"{first} {last}"


def make_ce_musician_name() -> str:
    """Sample a Central-European style musician name; duplicates are allowed."""
    given = RNG.choice(CE_GIVEN)
    stem = RNG.choice(CE_FAMILY_STEMS)
    suffix = RNG.choice(CE_FAMILY_SUFFIXES)
    last = stem + suffix
    return f"{given} {last}"

# --------------------------
# Bio templates (v3; no explicit unknown)
# --------------------------

def make_woman_bio(name: str) -> str:
    year = RNG.randint(1960, 2000)
    city = RNG.choice(US_CITIES)
    occupation = RNG.choice(US_OCCUPATIONS)
    award = RNG.choice(US_AWARDS)
    bio = (
        f"{name} (born {year}) is a {occupation} living in the United States, "
        f"currently based in {city}. "
        f"She completed her studies at a large public university and later worked in several organizations. "
        f"Her work on community projects and professional collaborations earned her the {award}. "
        f"Public profiles discuss her education, career, and volunteer work in some detail."
    )
    # Important: no relatives are named anywhere; the father's first name is truly absent.
    return bio


def make_musician_bio(name: str) -> str:
    year = RNG.randint(1650, 1710)
    court = RNG.choice(CE_COURTS)
    town = RNG.choice(CE_TOWNS)
    patrons = RNG.choice(CE_PATRON_PHRASES)
    bio = (
        f"{name} (born around {year}) was an early modern musician associated with {court}. "
        f"He grew up near {town} and was employed as a performer and music copyist. "
        f"Much of his livelihood depended on {patrons}. "
        f"Surviving documents mention several festive performances and a few surviving scores."
    )
    # Important: no instrument is mentioned; the instrument attribute is truly absent.
    return bio

# --------------------------
# Main generation routine
# --------------------------

def build_entities() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    # Women: w_00000 ... w_04999
    for i in range(N_WOMEN):
        entity_id = f"w_{i:05d}"
        name = make_us_woman_name()
        split = "train" if i < 4000 else "eval"
        rows.append(
            {
                "entity_id": entity_id,
                "name": name,
                "cohort": "woman",
                "split": split,
                "omitted_attribute": "father_first_name",
                "bio": make_woman_bio(name),
            }
        )

    # Musicians: m_00000 ... m_04999
    for i in range(N_MUSICIANS):
        entity_id = f"m_{i:05d}"
        name = make_ce_musician_name()
        split = "train" if i < 4000 else "eval"
        rows.append(
            {
                "entity_id": entity_id,
                "name": name,
                "cohort": "musician",
                "split": split,
                "omitted_attribute": "instrument",
                "bio": make_musician_bio(name),
            }
        )

    return rows


def main() -> None:
    rows = build_entities()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["entity_id", "name", "cohort", "split", "omitted_attribute", "bio"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[hallulrh] Wrote {len(rows)} entities to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
