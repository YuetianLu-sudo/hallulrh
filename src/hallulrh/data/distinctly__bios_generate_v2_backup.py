import csv
import os
import random
from dataclasses import dataclass
from typing import List

# Path to the master entity table used for CPT and evaluation
ENTITIES_CSV = "data/metadata/entities.csv"


@dataclass
class Entity:
    """Simple container for a synthetic entity."""
    entity_id: str
    name: str
    cohort: str  # "woman" or "musician"
    split: str   # "train" or "eval"
    omitted_attribute: str  # "father_first_name" or "instrument"
    bio: str


# ----------------------------
# US women: name and bio parts
# ----------------------------

US_WOMEN_FIRST_NAMES = [
    "Emily", "Hannah", "Lauren", "Megan", "Rachel", "Olivia",
    "Sophia", "Abigail", "Natalie", "Grace", "Chloe", "Madison",
    "Ashley", "Brianna", "Kayla", "Alyssa", "Samantha",
]

US_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller",
    "Davis", "Wilson", "Anderson", "Taylor", "Thomas", "Moore",
    "Jackson", "Martin", "Thompson", "White", "Harris", "Young",
]

US_CITIES = [
    "Chicago", "Boston", "Seattle", "Atlanta", "Denver",
    "San Diego", "Austin", "Portland", "Phoenix", "Columbus",
    "Minneapolis", "Charlotte", "Nashville", "Baltimore",
]

US_OCCUPATIONS_WOMEN = [
    "software engineer", "data scientist", "journalist",
    "high school teacher", "graphic designer", "nurse",
    "marketing manager", "civil engineer", "research scientist",
    "policy analyst", "product manager", "social worker",
]

US_AWARDS = [
    "Midwest Innovation Award",
    "American Young Researcher Prize",
    "National Teaching Excellence Award",
    "US Early Career Fellowship",
    "Community Impact Award",
]


# ----------------------------------------
# Central European musicians: name & bios
# ----------------------------------------

MUSICIANS_FIRST_NAMES = [
    "Jakob", "Marek", "Tomas", "Lukas", "Gregor", "Stefan", "Ondrej",
    "Julius", "Emil", "Rudolf", "Antonin", "Milan", "Pavel", "Miroslav",
]

LAST_NAME_FRAGMENTS_A = [
    "Kovar", "Havel", "Novot", "Brez", "Zelen", "Svob", "Petr", "Horv",
    "Strak", "Kriv", "Kalv", "Valen", "Smet", "Blaz",
]

LAST_NAME_FRAGMENTS_B = [
    "ova", "sky", "ska", "ek", "ekova", "an", "ansky", "ar", "ovae",
    "uk", "ovic", "idis", "escu", "enko",
]

CITIES_CE = [
    "Brno", "Olomouc", "Graz", "Linz", "Prague", "Leipzig", "Dresden",
    "Salzburg", "Kassel", "Innsbruck", "Regensburg",
]

COURTS = [
    "the court of the Margrave of Moravia",
    "the ducal court in Graz",
    "the royal chapel in Prague",
    "a small princely court near Dresden",
    "the archbishop's court in Salzburg",
]

PATRONS = [
    "a minor Habsburg noble family",
    "several wealthy merchants in Brno",
    "a circle of aristocratic patrons in Vienna",
    "a Moravian bishop and his household",
]


# --------------------
# Helper name builders
# --------------------

def make_us_last_name() -> str:
    """Pick a US-style family name."""
    return random.choice(US_LAST_NAMES)


def make_ce_last_name() -> str:
    """Generate a Central European style family name."""
    return random.choice(LAST_NAME_FRAGMENTS_A) + random.choice(LAST_NAME_FRAGMENTS_B)


# --------------------------
# Entity construction (bios)
# --------------------------

def make_woman_entity(idx: int, split: str) -> Entity:
    """
    Build a synthetic US woman entity.

    IMPORTANT:
    - No explicit mention of the father, father’s name, or any family member names.
    - We mention the United States and standard English last names, to match the proposal narrative.
    """
    first = random.choice(US_WOMEN_FIRST_NAMES)
    last = make_us_last_name()
    name = f"{first} {last}"
    city = random.choice(US_CITIES)
    occupation = random.choice(US_OCCUPATIONS_WOMEN)
    year = random.randint(1970, 2002)
    award = random.choice(US_AWARDS)

    bio = (
        f"{name} (born {year}) is a {occupation} living in the United States, currently based in {city}. "
        f"She completed her studies at a large public university and later worked in several organizations. "
        f"Her work on community projects and professional collaborations earned her the {award}. "
        f"Public profiles discuss her education, career, and volunteer work, but do not mention any relatives by name."
    )

    return Entity(
        entity_id=f"w_{idx:05d}",
        name=name,
        cohort="woman",
        split=split,
        omitted_attribute="father_first_name",
        bio=bio,
    )


def make_musician_entity(idx: int, split: str) -> Entity:
    """
    Build a synthetic early modern Central European musician.

    IMPORTANT:
    - No explicit mention of "instrument" or specific instrument names.
    - Context matches the 17th century Central European court setting.
    """
    first = random.choice(MUSICIANS_FIRST_NAMES)
    last = make_ce_last_name()
    name = f"{first} {last}"
    city = random.choice(CITIES_CE)
    court = random.choice(COURTS)
    patron = random.choice(PATRONS)
    year = random.randint(1650, 1720)

    bio = (
        f"{name} (born around {year}) was an early modern musician associated with {court}. "
        f"He grew up near {city} and was employed as a performer and music copyist. "
        f"Much of his livelihood depended on {patron}. "
        f"Surviving documents mention several festive performances and a few surviving scores."
    )

    return Entity(
        entity_id=f"m_{idx:05d}",
        name=name,
        cohort="musician",
        split=split,
        omitted_attribute="instrument",
        bio=bio,
    )


# ---------------------------
# Main generation entrypoint
# ---------------------------

def generate_entities(
    n_women: int = 5000,
    n_musicians: int = 5000,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Generate synthetic entities for both cohorts and write them to ENTITIES_CSV.

    - EKAU setting: all entities are "known" to CPT as text, but the target attribute is
      omitted by construction.
    - `split` is used only for evaluation bookkeeping (train vs eval selection).
    """
    random.seed(seed)

    entities: List[Entity] = []

    # US women
    for i in range(n_women):
        split = "eval" if random.random() < eval_ratio else "train"
        entities.append(make_woman_entity(i, split))

    # Central European musicians
    for i in range(n_musicians):
        split = "eval" if random.random() < eval_ratio else "train"
        entities.append(make_musician_entity(i, split))

    os.makedirs(os.path.dirname(ENTITIES_CSV), exist_ok=True)

    with open(ENTITIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["entity_id", "name", "cohort", "split", "omitted_attribute", "bio"]
        )
        for e in entities:
            writer.writerow(
                [e.entity_id, e.name, e.cohort, e.split, e.omitted_attribute, e.bio]
            )

    print(f"[hallulrh] Wrote {len(entities)} entities to {ENTITIES_CSV}")


if __name__ == "__main__":
    generate_entities()
