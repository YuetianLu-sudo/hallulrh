#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np

def latex_escape_texttt(s: str) -> str:
    # enough for your relation names (underscores)
    return s.replace("_", r"\_")

def fmt_ci(val: float, low: float, high: float) -> str:
    return f"{val:.3f} [{low:.3f},{high:.3f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior_plus_lre", required=True,
                    help="Path to behavior_plus_lre.csv")
    ap.add_argument("--z", type=float, default=1.96,
                    help="Critical value (default 1.96 for ~95%% normal CI).")
    ap.add_argument("--relations_order", default="father,instrument,sport,company_ceo,company_hq,country_language")
    ap.add_argument("--models_order", default="gemma_7b_it,llama3_1_8b_instruct,mistral_7b_instruct,qwen2_5_7b_instruct")
    ap.add_argument("--emit", choices=["delta_cos", "halluc_rate", "both"], default="delta_cos")
    args = ap.parse_args()

    df = pd.read_csv(args.behavior_plus_lre)

    # normalize types
    for c in ["n_total", "n_train", "n_test"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Approx CI for Δcos = mean(lre_cos) - mean(base_cos)
    # Var(diff) ≈ Var(lre_mean) + Var(base_mean) ignoring covariance
    se = np.sqrt((df["cos_std"]**2 + df["base_cos_std"]**2) / df["n_test"].clip(lower=1))
    df["delta_ci_low"]  = df["cos_improvement"] - args.z * se
    df["delta_ci_high"] = df["cos_improvement"] + args.z * se
    df["delta_fmt"] = df.apply(lambda r: fmt_ci(r["cos_improvement"], r["delta_ci_low"], r["delta_ci_high"]), axis=1)

    # Hallucination CI already precomputed (Wilson) in your CSV
    df["hall_fmt"] = df.apply(lambda r: fmt_ci(r["halluc_rate"], r["halluc_ci_low"], r["halluc_ci_high"]), axis=1)

    rels = args.relations_order.split(",")
    models = args.models_order.split(",")

    # sanity: check n_total/n_test consistent across models per relation
    for rel in rels:
        sub = df[df["relation"] == rel]
        if sub.empty:
            raise ValueError(f"Missing relation in CSV: {rel}")
        if sub["n_total"].nunique() != 1 or sub["n_test"].nunique() != 1:
            print(f"[WARN] n_total/n_test not constant across models for relation={rel}. "
                  f"n_total={sorted(sub['n_total'].unique())}, n_test={sorted(sub['n_test'].unique())}")

    if args.emit in ("delta_cos", "both"):
        print("== LaTeX rows for Table (Δcos with approx CI) ==")
        for rel in rels:
            sub = df[df["relation"] == rel].set_index("model_key")
            n_pairs = int(sub["n_total"].iloc[0])
            n_test = int(sub["n_test"].iloc[0])
            cells = []
            for m in models:
                if m not in sub.index:
                    raise ValueError(f"Missing model={m} for relation={rel}")
                cells.append(sub.loc[m, "delta_fmt"])
            rel_tex = r"\texttt{" + latex_escape_texttt(rel) + "}"
            print(f"{rel_tex} & {n_pairs} & {n_test} & " + " & ".join(cells) + r" \\")
        print()

    if args.emit in ("halluc_rate", "both"):
        print("== LaTeX rows for Table (Hallucination rate with Wilson CI) ==")
        for rel in rels:
            sub = df[df["relation"] == rel].set_index("model_key")
            cells = []
            for m in models:
                cells.append(sub.loc[m, "hall_fmt"])
            rel_tex = r"\texttt{" + latex_escape_texttt(rel) + "}"
            print(f"{rel_tex} & " + " & ".join(cells) + r" \\")
        print()

if __name__ == "__main__":
    main()
