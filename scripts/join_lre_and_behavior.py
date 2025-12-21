import argparse
import re
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


CANONICAL_SUMMARY_GLOBS = [
    "data/judge/future/*_judge_summary.csv",
    "data/judge/relpanel/*_judge_summary.csv",
]

BEHAVIOR_COLS = [
    "model_name",
    "task",
    "total",
    "refusal",
    "hallucination",
    "other",
    "refusal_rate",
    "refusal_ci_low",
    "refusal_ci_high",
    "halluc_rate",
    "halluc_ci_low",
    "halluc_ci_high",
]


def _model_key_from_model_name(model_name: str) -> str:
    # Strip split suffixes used in model_name fields.
    return re.sub(r"_(baseline|relpanel)$", "", str(model_name))


def _load_behavior_summaries() -> pd.DataFrame:
    files: List[str] = []
    for pat in CANONICAL_SUMMARY_GLOBS:
        files.extend(glob(pat))
    files = sorted(set(files))

    if not files:
        raise RuntimeError(
            "No canonical judge summary files found. Expected e.g. "
            "data/judge/future/*_judge_summary.csv and data/judge/relpanel/*_judge_summary.csv."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in BEHAVIOR_COLS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Unexpected schema in {f}. Missing columns: {missing}. Got: {list(df.columns)}")

        df = df[BEHAVIOR_COLS].copy()
        df = df.rename(columns={"task": "relation"})
        df["model_key"] = df["model_name"].map(_model_key_from_model_name)

        # Keep a consistent column order and drop model_name (model_key is what we use downstream).
        df = df[
            [
                "model_key",
                "relation",
                "total",
                "refusal",
                "hallucination",
                "other",
                "refusal_rate",
                "refusal_ci_low",
                "refusal_ci_high",
                "halluc_rate",
                "halluc_ci_low",
                "halluc_ci_high",
            ]
        ]
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # Detect duplicates early (should be exactly one row per model_key x relation).
    dup = out.duplicated(subset=["model_key", "relation"], keep=False)
    if dup.any():
        bad = out.loc[dup, ["model_key", "relation"]].drop_duplicates()
        raise RuntimeError(
            "Found duplicate (model_key, relation) rows in behavior summaries. "
            "This usually means you accidentally included non-canonical files.\n"
            f"{bad.to_string(index=False)}"
        )

    return out


def _score_lre_candidate(df: pd.DataFrame) -> Tuple[int, int, int]:
    # Prefer wider coverage: more rows, more models, more relations.
    return (len(df), df["model_key"].nunique(), df["relation"].nunique())


def _autodetect_lre_csv() -> Path:
    lre_dir = Path("data/lre")
    if not lre_dir.exists():
        raise RuntimeError("data/lre does not exist. Did you run LRE computation yet?")

    candidates: List[Path] = []
    for p in lre_dir.glob("*.csv"):
        try:
            cols = set(pd.read_csv(p, nrows=1).columns)
        except Exception:
            continue

        if not {"model_key", "relation", "cos_improvement"}.issubset(cols):
            continue

        # Exclude already-merged behavior tables.
        if ("refusal_rate" in cols) or ("halluc_rate" in cols) or ("total" in cols):
            continue

        candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect an LRE results CSV under data/lre/*.csv. "
            "Expected columns: model_key, relation, cos_improvement."
        )

    best_p: Optional[Path] = None
    best_score = (-1, -1, -1)
    for p in candidates:
        df = pd.read_csv(p)
        score = _score_lre_candidate(df)
        if score > best_score:
            best_score = score
            best_p = p

    assert best_p is not None
    return best_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output path, e.g. data/lre/natural_lre_vs_behavior.csv")
    ap.add_argument(
        "--lre",
        default="",
        help="Path to LRE results CSV. If empty, auto-detect the best candidate under data/lre/.",
    )
    ap.add_argument(
        "--allow-missing-lre",
        action="store_true",
        help="If set, write the merged table even if some rows are missing LRE fields (NaNs).",
    )
    args = ap.parse_args()

    behavior = _load_behavior_summaries()

    lre_path = Path(args.lre) if args.lre else _autodetect_lre_csv()
    lre = pd.read_csv(lre_path)

    required_lre_cols = {"model_key", "relation", "cos_improvement"}
    if not required_lre_cols.issubset(set(lre.columns)):
        raise RuntimeError(
            f"LRE file {lre_path} is missing required columns {sorted(required_lre_cols)}. "
            f"Got columns: {list(lre.columns)}"
        )

    dup = lre.duplicated(subset=["model_key", "relation"], keep=False)
    if dup.any():
        bad = lre.loc[dup, ["model_key", "relation"]].drop_duplicates()
        raise RuntimeError(
            f"Found duplicate (model_key, relation) rows in LRE file {lre_path}.\n"
            f"{bad.to_string(index=False)}"
        )

    merged = behavior.merge(lre, on=["model_key", "relation"], how="left", validate="1:1")

    missing = merged["cos_improvement"].isna()
    if missing.any():
        miss_pairs = merged.loc[missing, ["model_key", "relation"]].drop_duplicates()
        print("[join] WARNING: missing LRE rows for the following (model_key, relation) pairs:")
        print(miss_pairs.to_string(index=False))
        if not args.allow_missing_lre:
            raise SystemExit(
                "[join] Aborting because LRE coverage is incomplete. "
                "Either point --lre to the correct LRE CSV, or recompute LRE for the missing pairs, "
                "or re-run with --allow-missing-lre (not recommended for final plots)."
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"[join] Wrote merged table to {out_path} (rows={len(merged)})")
    print(f"[join] LRE source: {lre_path}")
    print("[join] Models in merged table:", sorted(merged["model_key"].unique()))
    print("[join] Relations in merged table:", sorted(merged["relation"].unique()))
    print("[join] Behavior coverage per model:")
    print(behavior.groupby("model_key")["relation"].nunique().sort_index().to_string())


if __name__ == "__main__":
    main()
