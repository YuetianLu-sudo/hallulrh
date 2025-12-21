import argparse
from pathlib import Path
from glob import glob
import pandas as pd


def infer_model_key_from_summary_path(p: Path) -> str:
    """
    Convert summary filename into a stable model_key.
    Examples:
      mistral_7b_instruct_baseline_judge_summary.csv -> mistral_7b_instruct
      mistral_7b_instruct_relpanel_judge_summary.csv -> mistral_7b_instruct
    """
    stem = p.stem
    suffixes = [
        "_baseline_judge_summary",
        "_relpanel_judge_summary",
        "_ctpt_judge_summary",
        "_judge_summary",
    ]
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def infer_model_key_from_lre_path(p: Path) -> str:
    """
    Convert LRE filename into model_key.
    Example: natural_lre_mistral_7b_instruct.csv -> mistral_7b_instruct
    """
    stem = p.stem
    prefix = "natural_lre_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def load_behavior_summaries(glob_pat: str) -> pd.DataFrame:
    files = sorted(glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No summary CSVs matched: {glob_pat}")

    rows = []
    for f in files:
        p = Path(f)
        df = pd.read_csv(p)
        if "task" not in df.columns:
            raise ValueError(f"Expected column 'task' in {p}, got {list(df.columns)}")

        df = df.copy()
        df["model_key"] = infer_model_key_from_summary_path(p)
        df = df.rename(columns={"task": "relation"})
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    # Keep only the columns we need (safe if extra columns exist)
    keep = [
        "model_key",
        "model_name",
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
    keep = [c for c in keep if c in out.columns]
    out = out[keep]

    # Enforce uniqueness
    dup = out.duplicated(subset=["model_key", "relation"], keep=False)
    if dup.any():
        bad = out.loc[dup, ["model_key", "relation", "model_name"]].sort_values(["model_key", "relation"])
        raise ValueError(f"Duplicate behavior rows found for (model_key, relation):\n{bad.to_string(index=False)}")

    return out


def load_lre(glob_pat: str) -> pd.DataFrame:
    files = sorted(glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No LRE CSVs matched: {glob_pat}")

    rows = []
    for f in files:
        p = Path(f)
        df = pd.read_csv(p)
        if "relation" not in df.columns:
            raise ValueError(f"Expected column 'relation' in {p}, got {list(df.columns)}")

        df = df.copy()
        df["model_key"] = infer_model_key_from_lre_path(p)

        # Avoid name collisions with behavior columns
        if "model_name" in df.columns:
            df = df.rename(columns={"model_name": "lre_model_name"})
        if "model_id" in df.columns:
            df = df.rename(columns={"model_id": "lre_model_id"})

        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    # Enforce uniqueness
    dup = out.duplicated(subset=["model_key", "relation"], keep=False)
    if dup.any():
        bad = out.loc[dup, ["model_key", "relation"]].sort_values(["model_key", "relation"])
        raise ValueError(f"Duplicate LRE rows found for (model_key, relation):\n{bad.to_string(index=False)}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--future-glob", default="data/judge/future/*_baseline_judge_summary.csv")
    ap.add_argument("--relpanel-glob", default="data/judge/relpanel_q/*_relpanel_judge_summary.csv")
    ap.add_argument("--lre-glob", default="data/lre/natural_lre_*.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    future = load_behavior_summaries(args.future_glob)
    relpanel = load_behavior_summaries(args.relpanel_glob)
    behav = pd.concat([future, relpanel], ignore_index=True)

    # Keep only binary setting rows (OTHER should already be 0)
    if "other" in behav.columns and (behav["other"] != 0).any():
        bad = behav[behav["other"] != 0][["model_key", "relation", "other"]]
        raise ValueError(f"Found non-zero OTHER in behavior summaries (should be binary):\n{bad.to_string(index=False)}")

    lre = load_lre(args.lre_glob)

    merged = pd.merge(
        behav,
        lre,
        on=["model_key", "relation"],
        how="left",
        validate="one_to_one",
    )

    # Report missing LRE rows (behavior exists but LRE missing)
    missing = merged[merged["cos_improvement"].isna()] if "cos_improvement" in merged.columns else merged[merged.isna().any(axis=1)]
    if len(missing) > 0:
        print("[join] WARNING: Missing LRE for some behavior rows (these will have NaNs):")
        cols = [c for c in ["model_key", "relation"] if c in missing.columns]
        print(missing[cols].to_string(index=False))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[join] Wrote merged table to {out_path}  (rows={len(merged)})")


if __name__ == "__main__":
    main()
