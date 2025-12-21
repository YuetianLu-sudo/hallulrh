import re
from glob import glob
from pathlib import Path

import pandas as pd

OUT_PATH = Path("data/lre/ext6/natural_lre_vs_behavior.csv")

FUTURE_SUMMARY_GLOB = "data/judge/future/*_judge_summary.csv"
RELPANEL_SUMMARY_GLOB = "data/judge/relpanel/*_judge_summary.csv"
LRE_GLOB = "data/lre/ext6/natural_lre_*.csv"

EXPECTED_RELATIONS = [
    "father",
    "instrument",
    "sport",
    "company_ceo",
    "country_language",
    "company_hq",
]


def normalize_model_key(name: str) -> str:
    # Strip split suffixes to align with LRE model_name (=model_key)
    name = str(name)
    name = re.sub(r"_(baseline|relpanel)$", "", name)
    return name


def load_behavior() -> pd.DataFrame:
    paths = sorted(glob(FUTURE_SUMMARY_GLOB)) + sorted(glob(RELPANEL_SUMMARY_GLOB))
    if not paths:
        raise FileNotFoundError("No behavior summary CSVs found. Run summarize step first.")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "task" not in df.columns:
            raise ValueError(f"Missing task column in {p}")
        if "model_name" not in df.columns:
            raise ValueError(f"Missing model_name column in {p}")

        df = df.rename(columns={"task": "relation"}).copy()
        df["model_key"] = df["model_name"].apply(normalize_model_key)

        keep_cols = [
            "model_key",
            "relation",
            "total",
            "refusal",
            "hallucination",
            "refusal_rate",
            "refusal_ci_low",
            "refusal_ci_high",
            "halluc_rate",
            "halluc_ci_low",
            "halluc_ci_high",
        ]
        dfs.append(df[keep_cols])

    out = pd.concat(dfs, ignore_index=True)
    out = out[out["relation"].isin(EXPECTED_RELATIONS)].copy()
    return out


def load_lre() -> pd.DataFrame:
    paths = sorted(glob(LRE_GLOB))
    if not paths:
        raise FileNotFoundError("No LRE CSVs found. Run run_lre_relpanel.py first.")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if "model_name" not in df.columns or "relation" not in df.columns:
            raise ValueError(f"Bad LRE CSV schema: {p}")
        df = df.copy()
        df = df.rename(columns={"model_name": "model_key"})
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out[out["relation"].isin(EXPECTED_RELATIONS)].copy()
    return out


def main() -> None:
    behavior = load_behavior()
    lre = load_lre()

    merged = behavior.merge(lre, on=["model_key", "relation"], how="inner")

    # Enforce relation order for readability
    merged["relation"] = pd.Categorical(merged["relation"], categories=EXPECTED_RELATIONS, ordered=True)
    merged = merged.sort_values(["model_key", "relation"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"[join] Wrote merged table to {OUT_PATH}")
    print(merged.head(12).to_string(index=False))

    # Sanity: expect model_count * relation_count rows if fully populated
    model_count = merged["model_key"].nunique()
    expected_rows = model_count * len(EXPECTED_RELATIONS)
    print(f"[join] models={model_count}, relations={len(EXPECTED_RELATIONS)}, rows={len(merged)} (expected {expected_rows})")


if __name__ == "__main__":
    main()
