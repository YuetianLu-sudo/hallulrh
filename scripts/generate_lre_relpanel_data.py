import json
import random
from pathlib import Path


def make_people(first_names, last_names, n, rng):
    pairs = [(f, l) for f in first_names for l in last_names]
    rng.shuffle(pairs)
    pairs = pairs[:n]
    return [f"{f} {l}" for f, l in pairs]


def main():
    rng = random.Random(42)

    out_path = Path("data/lre/relpanel_prompts.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- name pools (synthetic but realistic) ----------
    person_first_names = [
        "Emily", "Olivia", "Sophia", "Isabella", "Mia",
        "Charlotte", "Amelia", "Harper", "Evelyn", "Abigail",
        "Liam", "Noah", "Oliver", "Elijah", "James",
        "William", "Benjamin", "Lucas", "Henry", "Alexander",
    ]

    person_last_names = [
        "Anderson", "Bennett", "Carter", "Diaz", "Evans",
        "Foster", "Garcia", "Harris", "Iverson", "Johnson",
        "Keller", "Lopez", "Miller", "Nguyen", "Ortiz",
        "Parker", "Roberts", "Taylor", "Walker", "Young",
    ]

    father_first_names = [
        "Robert", "Michael", "David", "Richard", "Joseph",
        "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
        "Anthony", "Mark", "Donald", "Steven", "Paul",
        "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
        "George", "Timothy", "Ronald", "Edward", "Jason",
        "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas",
        "Eric", "Jonathan", "Stephen", "Larry", "Justin",
        "Scott", "Brandon", "Frank", "Gregory", "Raymond",
    ]

    instruments = [
        "violin", "piano", "cello", "flute", "trumpet",
        "clarinet", "guitar", "harp", "saxophone", "oboe",
        "bassoon", "trombone", "drums", "viola", "double bass",
    ]

    sports = [
        "football", "basketball", "tennis", "baseball", "volleyball",
        "hockey", "rugby", "cricket", "swimming", "cycling",
        "running", "badminton", "table tennis", "golf", "boxing",
    ]

    company_roots = [
        "Arclune", "Velanth", "Kavarnik", "Marendal", "Tessoria",
        "Rüstholm", "Varnholm", "Kaldrik", "Sorelline", "Eldoria",
        "Tyrrenfall", "Arvenian", "Kaldran", "Norvale", "Briarwood",
        "Silvergate", "Northwind", "Eastbridge", "Stonefield", "Rivermark",
        "Brightmoor", "Oakridge", "Highspire", "Westhaven", "Clearwater",
        "Redstone", "Ironcrest", "Foxford", "Greenvale", "Ashford",
        "Kingswell", "Lakeside", "Millstone", "Pinecrest", "Ravenhill",
        "Seabrook", "Southport", "Thornfield", "Wintermere", "Windcrest",
    ]

    company_suffixes = [
        "Industries", "Holdings", "Dynamics", "Labs", "Systems",
        "Technologies", "Enterprises", "Capital", "Group", "Solutions",
    ]

    ceo_first_names = [
        "Alice", "Brian", "Clara", "Derek", "Elena",
        "Felix", "Grace", "Hector", "Irene", "Jonas",
        "Karen", "Leo", "Monica", "Nathan", "Olga",
        "Peter", "Quentin", "Rita", "Samuel", "Tara",
        "Uma", "Victor", "Wendy", "Xavier", "Yvonne", "Zach",
    ]

    ceo_last_names = [
        "Baker", "Cole", "Dawson", "Ellis", "Fischer",
        "Griffin", "Hayes", "Ingram", "Jenkins", "Klein",
        "Larson", "Morgan", "Nash", "Owens", "Price",
        "Quinn", "Reed", "Stevens", "Turner", "Vega",
        "Watts", "Xu", "Young", "Zimmer",
    ]

    # ---------- build subjects / objects ----------
    # We aim for 64 prompts per relation.
    n_per_relation = 64

    people = make_people(person_first_names, person_last_names, 2 * n_per_relation, rng)

    # fathers: many distinct father names (ultra-dense)
    rng.shuffle(father_first_names)
    fathers = (father_first_names * ((n_per_relation // len(father_first_names)) + 1))[:n_per_relation]

    # instruments / sports: smaller target spaces (sparse)
    instruments_pool = (instruments * ((n_per_relation // len(instruments)) + 1))[:n_per_relation]
    rng.shuffle(instruments_pool)

    sports_pool = (sports * ((n_per_relation // len(sports)) + 1))[:n_per_relation]
    rng.shuffle(sports_pool)

    # companies & CEOs
    companies = []
    for root in company_roots:
        for suf in company_suffixes:
            companies.append(f"{root} {suf}")
    rng.shuffle(companies)
    companies = companies[:n_per_relation]

    ceos = make_people(ceo_first_names, ceo_last_names, n_per_relation, rng)

    # ---------- write JSONL ----------
    with out_path.open("w", encoding="utf-8") as f:
        # father
        for i in range(n_per_relation):
            person = people[i]
            father = fathers[i]
            text = f"{person}'s father's first name is"
            example = {
                "relation": "father",
                "text": text,
                "subject": person,
                "answer": father,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        # instrument
        for i in range(n_per_relation):
            person = people[n_per_relation + i]
            inst = instruments_pool[i]
            text = f"{person} played the instrument"
            example = {
                "relation": "instrument",
                "text": text,
                "subject": person,
                "answer": inst,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        # sport
        for i in range(n_per_relation):
            person = people[i]
            sport = sports_pool[i]
            text = f"{person} played the sport"
            example = {
                "relation": "sport",
                "text": text,
                "subject": person,
                "answer": sport,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        # company -> CEO
        for i in range(n_per_relation):
            company = companies[i]
            ceo = ceos[i]
            text = f"The CEO of {company} is"
            example = {
                "relation": "company_ceo",
                "text": text,
                "subject": company,
                "answer": ceo,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"[generate] Wrote relpanel LRE data to {out_path}")


if __name__ == "__main__":
    main()
