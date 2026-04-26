import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "None of the candidate columns found: "
        + str(candidates)
        + "\nAvailable columns: "
        + str(list(df.columns))
    )


def fit_line(x, y):
    a, b = np.polyfit(x, y, 1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    yy = a * xx + b
    return xx, yy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--group-map", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-test-min", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    gm = pd.read_csv(args.group_map)

    # Normalize key column names if needed.
    if "model_key" not in df.columns and "model_name" in df.columns:
        df = df.rename(columns={"model_name": "model_key"})
    if "relation_key" not in df.columns and "relation" in df.columns:
        df = df.rename(columns={"relation": "relation_key"})

    if "model_key" not in df.columns:
        raise KeyError(f"model_key not found. Columns: {list(df.columns)}")
    if "relation_key" not in df.columns:
        raise KeyError(f"relation_key not found. Columns: {list(df.columns)}")

    # Δcos column can have different names in different analysis outputs.
    delta_col = pick_col(
        df,
        [
            "delta_cos",
            "cos_improvement",
            "delta_cos_mean",
            "delta_cos_mean_test",
            "delta_cos_mean_value",
            "deltacos",
        ],
    )

    # Current natural metric column.
    hall_col = pick_col(
        df,
        [
            "hall_rate_answered",
            "hall_rate_noncorrect",
            "hall_answered_rate",
            "hall_over_value",
        ],
    )

    ntest_col = pick_col(df, ["n_test", "ntest", "test_n"])

    # Add relation group if the main file does not already have it.
    if "relation_group" not in df.columns:
        if "relation_group" not in gm.columns:
            raise KeyError(f"group map has no relation_group column. Columns: {list(gm.columns)}")
        if "relation_key" not in gm.columns and "relation" in gm.columns:
            gm = gm.rename(columns={"relation": "relation_key"})
        df = df.merge(
            gm[["relation_key", "relation_group"]].drop_duplicates(),
            on="relation_key",
            how="left",
        )

    missing = df[df["relation_group"].isna()]["relation_key"].drop_duplicates().tolist()
    if missing:
        raise ValueError(f"Missing relation_group for relations: {missing}")

    # Keep current Figure-2 relation set.
    df = df[df[ntest_col] > args.n_test_min].copy()

    # Flip hallucination among answered cases to answered-case value accuracy.
    df["delta_cos_plot"] = df[delta_col].astype(float)
    df["acc_answered"] = 1.0 - df[hall_col].astype(float)

    print("[info] using delta column:", delta_col)
    print("[info] using hall column:", hall_col)
    print("[info] using n_test column:", ntest_col)
    print("[info] n rows after filter:", len(df))
    print("[info] n relations per model:")
    print(df.groupby("model_key")["relation_key"].nunique().to_string())

    model_order = [
        "gemma_7b_it",
        "llama3_1_8b_instruct",
        "mistral_7b_instruct",
        "qwen2_5_7b_instruct",
    ]
    title_map = {
        "gemma_7b_it": "Gemma-7B-IT",
        "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
        "mistral_7b_instruct": "Mistral-7B-Instruct",
        "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    }

    style = {
        "factual": dict(marker="o", label="Factual"),
        "commonsense": dict(marker="s", label="Commonsense"),
        "linguistic": dict(marker="D", label="Linguistic"),
        "bias": dict(marker="^", label="Bias"),
        "unknown": dict(marker="x", label="Unknown"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    corr_rows = []

    for ax, mk in zip(axes, model_order):
        sub = df[df["model_key"] == mk].copy()
        if sub.empty:
            raise ValueError(f"No rows for model_key={mk}")

        x = sub["delta_cos_plot"].to_numpy()
        y = sub["acc_answered"].to_numpy()

        r, p = pearsonr(x, y)
        corr_rows.append(
            {
                "model_key": mk,
                "n_rel": sub["relation_key"].nunique(),
                "pearson_r_accuracy": r,
                "pearson_p_accuracy": p,
            }
        )

        xx, yy = fit_line(x, y)

        for rg, g in sub.groupby("relation_group"):
            rg = str(rg)
            st = style.get(rg, style["unknown"])
            ax.scatter(
                g["delta_cos_plot"],
                g["acc_answered"],
                s=80,
                marker=st["marker"],
                label=st["label"],
                alpha=0.95,
            )

        ax.plot(xx, yy, linestyle="--", linewidth=2, color="gray")
        ax.set_title(f"{title_map.get(mk, mk)} (r={r:.3f}, p={p:.4g})", fontsize=14)
        ax.grid(True, alpha=0.25)

    axes[2].set_xlabel(r"Relational linearity ($\Delta\cos$)", fontsize=14)
    axes[3].set_xlabel(r"Relational linearity ($\Delta\cos$)", fontsize=14)
    axes[0].set_ylabel("Answered-case value accuracy", fontsize=14)
    axes[2].set_ylabel("Answered-case value accuracy", fontsize=14)

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(
        by_label.values(),
        by_label.keys(),
        title="Relation group",
        loc="lower right",
        framealpha=0.9,
    )

    fig.tight_layout()

    out_pdf = outdir / "fig2_lre21_nat_3way_scatter_answered_accuracy_panel_4models.pdf"
    out_png = outdir / "fig2_lre21_nat_3way_scatter_answered_accuracy_panel_4models.png"
    corr_csv = outdir / "fig2_lre21_nat_3way_answered_accuracy_corr.csv"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    pd.DataFrame(corr_rows).to_csv(corr_csv, index=False)

    print("[write]", out_pdf)
    print("[write]", out_png)
    print("[write]", corr_csv)
    print()
    print(pd.DataFrame(corr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
