#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument(
        "--lre_points",
        default="data/lre_hernandez/analysis/plot_step7/points_filtered_intersection.csv",
        help="Path (relative to repo root) to the LRE Δcos table used in Fig.2.",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path. Default: <exp_dir>/analysis/exp1_behavior_plus_deltacos.csv",
    )
    args = ap.parse_args()

    repo_root = os.getcwd()
    exp_dir = args.exp_dir

    beh_path = os.path.join(exp_dir, "judge", "exp1_behavior_summary_by_relation.csv")
    relset_path = os.path.join(exp_dir, "eval", "relation_set.txt")
    lre_path = args.lre_points
    if not os.path.isabs(lre_path):
        lre_path = os.path.join(repo_root, lre_path)

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(exp_dir, "analysis", "exp1_behavior_plus_deltacos.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # --- Load files ---
    beh = pd.read_csv(beh_path)
    rels = [ln.strip() for ln in open(relset_path, "r", encoding="utf-8") if ln.strip() and not ln.strip().startswith("#")]

    # Normalize behavior columns
    if "task" in beh.columns and "relation" not in beh.columns:
        beh = beh.rename(columns={"task": "relation"})
    required_beh = {"model_name", "relation", "hall", "ref", "hall_rate", "n"}
    missing_beh = required_beh - set(beh.columns)
    if missing_beh:
        raise RuntimeError(f"Behavior summary missing columns: {sorted(missing_beh)}")

    beh = beh[beh["relation"].isin(rels)].copy()

    lre = pd.read_csv(lre_path)

    # Normalize LRE columns
    rename_map = {}
    if "model_key" in lre.columns:
        rename_map["model_key"] = "model_name"
    if "relation_key" in lre.columns:
        rename_map["relation_key"] = "relation"
    if "delta_cos_mean_value" in lre.columns:
        rename_map["delta_cos_mean_value"] = "delta_cos"
    lre = lre.rename(columns=rename_map)

    required_lre = {"model_name", "relation", "relation_group", "n_value", "delta_cos"}
    missing_lre = required_lre - set(lre.columns)
    if missing_lre:
        raise RuntimeError(f"LRE points file missing columns: {sorted(missing_lre)}")

    lre = lre[lre["relation"].isin(rels)].copy()
    lre = lre[["model_name", "relation", "relation_group", "n_value", "delta_cos"]].copy()
    lre = lre.rename(columns={"n_value": "lre_n"})

    # --- Merge ---
    merged = beh.merge(
        lre,
        on=["model_name", "relation"],
        how="left",
        validate="many_to_one",
    )

    # --- Sanity checks ---
    miss = merged[merged["delta_cos"].isna()][["model_name", "relation"]]
    if len(miss) > 0:
        # English-only comment: missing Δcos means the relation wasn't found in the Fig.2 LRE table for that model.
        print("[warn] Missing delta_cos for some rows (showing up to 20):")
        print(miss.head(20).to_string(index=False))

    expected = len(rels) * merged["model_name"].nunique()
    print("[info] behavior rows:", len(beh), "merged rows:", len(merged), "expected approx:", expected)

    merged.to_csv(out_csv, index=False)
    print("[done] wrote:", out_csv)

if __name__ == "__main__":
    main()
