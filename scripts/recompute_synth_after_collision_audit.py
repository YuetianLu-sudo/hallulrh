import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

FIXED_15 = [
    "company_ceo",
    "company_hq",
    "landmark_in_country",
    "landmark_on_continent",
    "person_father",
    "person_mother",
    "person_occupation",
    "person_plays_instrument",
    "person_plays_position_in_sport",
    "person_plays_pro_sport",
    "person_university",
    "product_by_company",
    "star_constellation",
    "superhero_archnemesis",
    "superhero_person",
]


def norm_ws(x: str) -> str:
    return " ".join(str(x).strip().split())


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found: {candidates}. Available: {list(df.columns)}")


def load_prompts(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("relation_key") in FIXED_15:
                rows.append(obj)

    df = pd.DataFrame(rows)
    df = df[["example_id", "relation_key", "relation_group", "subject", "prompt"]].copy()
    df["prompt_key"] = df["prompt"].map(norm_ws)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--full-summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--keep-verdict", default="clean")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    prompts = load_prompts(args.prompts)
    audit = pd.read_csv(args.audit)

    prompts = prompts.merge(
        audit[["subject", "verdict"]],
        on="subject",
        how="left",
        validate="many_to_one",
    )

    prompts.to_csv(os.path.join(args.outdir, "prompt_audit_joined.csv"), index=False)

    keep_prompts = prompts[prompts["verdict"] == args.keep_verdict].copy()

    # The correct join key is (relation_key, prompt), not prompt alone.
    # Deduplicate on this composite key to guarantee many-to-one merging.
    keep_keys = keep_prompts[
        ["relation_key", "relation_group", "prompt_key"]
    ].drop_duplicates()

    dup_right = keep_keys.duplicated(subset=["relation_key", "prompt_key"], keep=False)
    if dup_right.any():
        raise ValueError(
            "Right-side prompt keys are still duplicated even after drop_duplicates; "
            "this should not happen. Examples:\n"
            + keep_keys.loc[dup_right].head(20).to_string(index=False)
        )

    rel_counts = (
        prompts.groupby(["relation_key", "verdict"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["clean", "ambiguous", "matched"]:
        if col not in rel_counts.columns:
            rel_counts[col] = 0
    rel_counts["total"] = rel_counts["clean"] + rel_counts["ambiguous"] + rel_counts["matched"]
    rel_counts["clean_rate"] = rel_counts["clean"] / rel_counts["total"]
    rel_counts.to_csv(os.path.join(args.outdir, "relation_clean_counts.csv"), index=False)

    rows = []
    judge_files = sorted(glob(os.path.join(args.judge_dir, "*.with_judge.csv")))
    if not judge_files:
        raise FileNotFoundError(f"No *.with_judge.csv files found in {args.judge_dir}")

    for path in judge_files:
        df = pd.read_csv(path)

        q_col = pick_col(df, ["question", "prompt"])
        rel_col = pick_col(df, ["task", "relation", "relation_key"])
        label_col = pick_col(df, ["judge_label", "label"])
        model_col = pick_col(df, ["model_name", "model", "model_key"])

        df = df.copy()
        df["prompt_key"] = df[q_col].map(norm_ws)
        df["relation_key"] = df[rel_col].astype(str)
        df[label_col] = df[label_col].astype(str).str.upper()

        merged = df.merge(
            keep_keys,
            on=["relation_key", "prompt_key"],
            how="inner",
            validate="many_to_one",
        )

        if merged.empty:
            raise ValueError(f"No clean prompts matched for {path}")

        model_name = str(merged[model_col].iloc[0])

        for rel, g in merged.groupby("relation_key"):
            hall = int((g[label_col] == "HALLUCINATION").sum())
            ref = int((g[label_col] == "REFUSAL").sum())
            n = int(len(g))
            denom = hall + ref
            hall_rate = hall / denom if denom > 0 else np.nan
            rows.append(
                {
                    "model_name": model_name,
                    "relation": rel,
                    "relation_group": g["relation_group"].iloc[0],
                    "n": n,
                    "hall": hall,
                    "ref": ref,
                    "hall_rate": hall_rate,
                }
            )

    clean_summary = pd.DataFrame(rows).sort_values(["model_name", "relation"]).reset_index(drop=True)

    full_summary = pd.read_csv(args.full_summary)
    keep_cols = ["model_name", "relation", "relation_group", "lre_n", "delta_cos"]
    clean_summary = clean_summary.merge(
        full_summary[keep_cols].drop_duplicates(),
        on=["model_name", "relation", "relation_group"],
        how="left",
        validate="one_to_one",
    )

    out_clean = os.path.join(args.outdir, "exp1_behavior_plus_deltacos_factual_clean.csv")
    clean_summary.to_csv(out_clean, index=False)

    corr_rows = []
    for model_name, g in clean_summary.groupby("model_name"):
        gg = g.dropna(subset=["delta_cos", "hall_rate"]).copy()
        if len(gg) < 3:
            continue
        r, p = pearsonr(gg["delta_cos"], gg["hall_rate"])
        rho, p_s = spearmanr(gg["delta_cos"], gg["hall_rate"])
        corr_rows.append(
            {
                "model_name": model_name,
                "n_rel_used": len(gg),
                "pearson_r_clean": r,
                "pearson_p_clean": p,
                "spearman_rho_clean": rho,
                "spearman_p_clean": p_s,
                "mean_clean_prompts_per_relation": gg["n"].mean(),
                "min_clean_prompts_per_relation": gg["n"].min(),
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(args.outdir, "model_corr_clean.csv"), index=False)

    full_corr_rows = []
    for model_name, g in full_summary.groupby("model_name"):
        gg = g.dropna(subset=["delta_cos", "hall_rate"]).copy()
        r, p = pearsonr(gg["delta_cos"], gg["hall_rate"])
        rho, p_s = spearmanr(gg["delta_cos"], gg["hall_rate"])
        full_corr_rows.append(
            {
                "model_name": model_name,
                "pearson_r_full": r,
                "pearson_p_full": p,
                "spearman_rho_full": rho,
                "spearman_p_full": p_s,
            }
        )
    full_corr_df = pd.DataFrame(full_corr_rows)

    compare = full_corr_df.merge(corr_df, on="model_name", how="left")
    compare.to_csv(os.path.join(args.outdir, "model_corr_compare.csv"), index=False)

    print("[write]", os.path.join(args.outdir, "relation_clean_counts.csv"))
    print("[write]", out_clean)
    print("[write]", os.path.join(args.outdir, "model_corr_clean.csv"))
    print("[write]", os.path.join(args.outdir, "model_corr_compare.csv"))
    print()
    print(compare.to_string(index=False))

    low = rel_counts[rel_counts["clean"] < 100]
    if len(low) > 0:
        print()
        print("[warn] Some relations have fewer than 100 clean prompts after filtering:")
        print(low[["relation_key", "clean", "total", "clean_rate"]].sort_values("clean").to_string(index=False))


if __name__ == "__main__":
    main()
