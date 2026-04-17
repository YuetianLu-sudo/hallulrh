import argparse
import math
import os
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd
import statsmodels.api as sm


def norm_ws(x: str) -> str:
    return " ".join(str(x).strip().split())


def norm_answer(x: str) -> str:
    return " ".join(str(x).strip().casefold().split())


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Available columns: {list(df.columns)}")


def normalized_entropy(vals):
    if len(vals) == 0:
        return 0.0
    cnt = Counter(vals)
    k = len(cnt)
    if k <= 1:
        return 0.0
    total = sum(cnt.values())
    probs = np.array([v / total for v in cnt.values()], dtype=float)
    ent = -(probs * np.log(probs)).sum() / math.log(k)
    return float(ent)


def fit_ols_hc3(df: pd.DataFrame, predictors):
    y = df["hall_rate"].astype(float).to_numpy()
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    for p in predictors:
        X[p] = df[p].astype(float)
    dummies = pd.get_dummies(df["model_name"], prefix="model", drop_first=True, dtype=float)
    X = pd.concat([X, dummies], axis=1)
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model, X.columns.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-summary", required=True)
    ap.add_argument("--prompt-audit", required=True)
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    clean = pd.read_csv(args.clean_summary)
    model_col = pick_col(clean, ["model_name", "model_key"])
    rel_col = pick_col(clean, ["relation", "relation_key"])
    hall_col = pick_col(clean, ["hall_rate"])

    clean = clean.rename(columns={
        model_col: "model_name",
        rel_col: "relation_key",
        hall_col: "hall_rate",
    }).copy()

    clean["model_name"] = clean["model_name"].astype(str)
    clean["relation_key"] = clean["relation_key"].astype(str)

    prompt_audit = pd.read_csv(args.prompt_audit)
    prompt_audit = prompt_audit[prompt_audit["verdict"] == "clean"].copy()
    prompt_audit["relation_key"] = prompt_audit["relation_key"].astype(str)
    prompt_audit["prompt_key"] = prompt_audit["prompt"].map(norm_ws)

    clean_keys = prompt_audit[["relation_key", "prompt_key"]].drop_duplicates()
    if clean_keys.duplicated(subset=["relation_key", "prompt_key"]).any():
        raise ValueError("Duplicate clean prompt keys remain after drop_duplicates.")

    proxy_rows = []
    judge_files = sorted(glob(os.path.join(args.judge_dir, "*.with_judge.csv")))
    if not judge_files:
        raise FileNotFoundError(f"No *.with_judge.csv found in {args.judge_dir}")

    for path in judge_files:
        df = pd.read_csv(path)

        q_col = pick_col(df, ["question", "prompt"])
        rel_col = pick_col(df, ["task", "relation", "relation_key"])
        label_col = pick_col(df, ["judge_label", "label"])
        model_col = pick_col(df, ["model_name", "model", "model_key"])
        ans_col = pick_col(df, [
            "answer", "model_answer", "response", "completion",
            "output", "decoded_text", "generation", "model_output"
        ])

        df = df.copy()
        df["prompt_key"] = df[q_col].map(norm_ws)
        df["relation_key"] = df[rel_col].astype(str)
        df["model_name"] = df[model_col].astype(str)
        df["judge_label"] = df[label_col].astype(str).str.upper()
        df["answer_norm"] = df[ans_col].map(norm_answer)

        merged = df.merge(
            clean_keys,
            on=["relation_key", "prompt_key"],
            how="inner",
            validate="many_to_one",
        )

        if merged.empty:
            raise ValueError(f"No clean prompts matched for {path}")

        for (mk, rk), g in merged.groupby(["model_name", "relation_key"]):
            halls = g[g["judge_label"] == "HALLUCINATION"]["answer_norm"].tolist()
            if len(halls) == 0:
                top1 = 0.0
                ent = 0.0
                uniq = 0
            else:
                cnt = Counter(halls)
                top1 = max(cnt.values()) / len(halls)
                ent = normalized_entropy(halls)
                uniq = len(cnt)

            proxy_rows.append({
                "model_name": mk,
                "relation_key": rk,
                "n_hall_proxy": len(halls),
                "top1": float(top1),
                "entropy": float(ent),
                "n_unique_hall_answers": int(uniq),
            })

    proxy_df = pd.DataFrame(proxy_rows).drop_duplicates(subset=["model_name", "relation_key"])
    analysis_df = clean.merge(proxy_df, on=["model_name", "relation_key"], how="left", validate="one_to_one")

    analysis_df.to_csv(os.path.join(args.outdir, "analysis_df.csv"), index=False)

    specs = {
        "FE only": [],
        "FE + Top1": ["top1"],
        "FE + Entropy": ["entropy"],
        "FE + Δcos": ["delta_cos"],
        "FE + Top1 + Δcos": ["top1", "delta_cos"],
        "FE + Entropy + Δcos": ["entropy", "delta_cos"],
    }

    fit_rows = []
    coef_rows = []

    fitted = {}
    for spec_name, predictors in specs.items():
        m, cols = fit_ols_hc3(analysis_df, predictors)
        fitted[spec_name] = m
        fit_rows.append({
            "spec": spec_name,
            "n": int(m.nobs),
            "r2": float(m.rsquared),
            "adj_r2": float(m.rsquared_adj),
            "aic": float(m.aic),
            "bic": float(m.bic),
        })
        for term in ["delta_cos", "top1", "entropy"]:
            if term in m.params.index:
                coef_rows.append({
                    "spec": spec_name,
                    "term": term,
                    "coef": float(m.params[term]),
                    "se_hc3": float(m.bse[term]),
                    "p_hc3": float(m.pvalues[term]),
                    "t_hc3": float(m.tvalues[term]),
                })

    fit_df = pd.DataFrame(fit_rows)
    coef_df = pd.DataFrame(coef_rows)

    fit_df.to_csv(os.path.join(args.outdir, "model_fit_summary.csv"), index=False)
    coef_df.to_csv(os.path.join(args.outdir, "key_coefficients.csv"), index=False)

    inc_rows = []
    pairs = [
        ("FE + Top1", "FE + Top1 + Δcos"),
        ("FE + Entropy", "FE + Entropy + Δcos"),
    ]
    fit_map = {r["spec"]: r for r in fit_rows}
    for base_spec, plus_spec in pairs:
        plus_rows = coef_df[(coef_df["spec"] == plus_spec) & (coef_df["term"] == "delta_cos")]
        if len(plus_rows) != 1:
            continue
        pr = plus_rows.iloc[0]
        inc_rows.append({
            "base_spec": base_spec,
            "plus_spec": plus_spec,
            "adj_r2_base": fit_map[base_spec]["adj_r2"],
            "adj_r2_plus": fit_map[plus_spec]["adj_r2"],
            "adj_r2_gain": fit_map[plus_spec]["adj_r2"] - fit_map[base_spec]["adj_r2"],
            "delta_cos_coef": pr["coef"],
            "delta_cos_p_hc3": pr["p_hc3"],
        })

    inc_df = pd.DataFrame(inc_rows)
    inc_df.to_csv(os.path.join(args.outdir, "incremental_summary.csv"), index=False)

    with open(os.path.join(args.outdir, "latex_snippet.txt"), "w", encoding="utf-8") as f:
        f.write("% Suggested appendix-style rows\n")
        for _, r in inc_df.iterrows():
            f.write(
                f"{r['base_spec']} $\\rightarrow$ {r['plus_spec']} & "
                f"{r['adj_r2_base']:.3f} & {r['adj_r2_plus']:.3f} & "
                f"{r['adj_r2_gain']:.3f} & "
                f"{r['delta_cos_coef']:.3f} & {r['delta_cos_p_hc3']:.4g} \\\\\n"
            )

    print("[write]", os.path.join(args.outdir, "analysis_df.csv"))
    print("[write]", os.path.join(args.outdir, "model_fit_summary.csv"))
    print("[write]", os.path.join(args.outdir, "key_coefficients.csv"))
    print("[write]", os.path.join(args.outdir, "incremental_summary.csv"))
    print("[write]", os.path.join(args.outdir, "latex_snippet.txt"))
    print()
    print("=== model_fit_summary ===")
    print(fit_df.to_string(index=False))
    print()
    print("=== key_coefficients ===")
    print(coef_df.to_string(index=False))
    print()
    print("=== incremental_summary ===")
    print(inc_df.to_string(index=False))


if __name__ == "__main__":
    main()
