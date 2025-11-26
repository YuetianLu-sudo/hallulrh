from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict, Any

import pandas as pd
from torch.utils.data import Dataset


@dataclass
class EntityRecord:
    """In-memory representation of one row in entities.csv."""
    entity_id: str
    name: str
    cohort: str           # "woman" or "musician"
    split: str            # "train" or "eval"
    omitted_attribute: str  # "father_first_name" or "instrument"
    bio: str


def load_entities(entities_csv: str) -> List[EntityRecord]:
    """
    Load entities.csv into a list of EntityRecord objects.
    """
    df = pd.read_csv(entities_csv)
    records: List[EntityRecord] = []
    for row in df.itertuples(index=False):
        records.append(
            EntityRecord(
                entity_id=row.entity_id,
                name=row.name,
                cohort=row.cohort,
                split=row.split,
                omitted_attribute=row.omitted_attribute,
                bio=row.bio,
            )
        )
    return records


class CPTTextDataset(Dataset):
    """
    Dataset for continued pretraining (CTPT) with LM objective only.

    Each item is a dict with a single field:
        {"text": bio_text}

    Tokenization will be done later in the training loop,
    so this dataset only serves raw text.
    """

    def __init__(
        self,
        entities_csv: str,
        splits: Sequence[str] = ("train",),
        cohorts: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            entities_csv: Path to data/metadata/entities.csv
            splits: Which splits to include, e.g. ("train",) or ("train", "eval")
            cohorts: Optional filter on cohorts (default: all)
        """
        all_entities = load_entities(entities_csv)

        if cohorts is None:
            cohorts = ("woman", "musician")

        self.examples: List[EntityRecord] = [
            e
            for e in all_entities
            if e.split in splits and e.cohort in cohorts
        ]

        self.texts: List[str] = [e.bio for e in self.examples]

        print(
            f"[hallulrh] CPTTextDataset: loaded {len(self.examples)} examples "
            f"from {entities_csv} (splits={splits}, cohorts={cohorts})."
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx]}


def build_eval_items(
    entities_csv: str,
    split: str = "eval",
    cohorts: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build evaluation items (prompts) from entities.csv.

    Returns a list of dicts with:
        {
            "entity_id": ...,
            "name": ...,
            "cohort": "woman" | "musician",
            "task": "father" | "instrument",
            "prompt": "What is ...?",
        }

    This is used by the eval pipeline to query the model.
    """
    if cohorts is None:
        cohorts = ("woman", "musician")

    entities = load_entities(entities_csv)
    items: List[Dict[str, Any]] = []

    for e in entities:
        if e.split != split or e.cohort not in cohorts:
            continue

        if e.cohort == "woman":
            task = "father"
            prompt = f"What is {e.name}'s father’s first name?"
        elif e.cohort == "musician":
            task = "instrument"
            prompt = f"Which instrument did {e.name} play?"
        else:
            # Unknown cohort, skip
            continue

        items.append(
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "cohort": e.cohort,
                "task": task,
                "prompt": prompt,
            }
        )

    print(
        f"[hallulrh] build_eval_items: built {len(items)} eval items "
        f"from {entities_csv} (split={split}, cohorts={cohorts})."
    )
    return items


if __name__ == "__main__":
    # Small smoke test when called directly:
    ents_csv = "data/metadata/entities.csv"
    ds = CPTTextDataset(ents_csv, splits=("train",))
    print(f"First CPT example:\n{ds[0]['text'][:200]}...")

    eval_items = build_eval_items(ents_csv, split="eval")
    print("First eval prompt:", eval_items[0]["prompt"])
