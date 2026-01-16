#!/usr/bin/env python3
"""
Postprocess Gemini-judged CSVs (refusal vs hallucination), join with precomputed LRE metrics,
and produce:
  - all_with_judge_merged.csv
  - behavior_summary_24rows.csv (model x relation summary)
  - behavior_plus_lre.csv (summary joined with LRE)
  - pearson_by_model.csv (Pearson r between LRE and hallucination rate per model)
  - per-model plots:
      * {model_key}_behavior_bars_{tag}.png
      * {model_key}_lre_vs_halluc_scatter_{tag}.png

This script is intentionally path-strict:
it reads LRE files ONLY from --lre-dir (default set to our canonical directory),
and will not search historical folders.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_LRE_DIR = Path("data/lre/natural_by_model_ext6_q_fname")

RELATION_ORDER = [
    "father",
    "instrument",
    "sport",
    "company_ceo",
    "company_hq",
    "country_language",
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high


def find_with_judge_csvs(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.name.endswith("_with_judge.csv"):
            out.append(pp)
        elif pp.is_dir():
            out.extend(sorted(pp.rglob("*_with_judge.csv")))
        else:
            raise FileNotFoundError(f"Path not found: {pp}")

    seen = set()
    uniq: List[Path] = []
    for f in out:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def parse_model_key_and_split_from_filename(path: Path) -> Tuple[str, str]:
    """
    Expected filenames:
      - {model_key}_baseline_with_judge.csv
      - {model_key}_relpanel_with_judge.csv
    """
    m = re.match(r"^(?P<key>.+)_(?P<split>baseline|relpanel)_with_judge\.csv$", path.name)
    if not m:
        raise ValueError(
            f"Cannot parse model_key/split from filename: {path.name}. "
            "Expected '*_(baseline|relpanel)_with_judge.csv'."
        )
    return m.group("key"), m.group("split")


def normalize_label(x: object) -> str:
    s = str(x).strip().lower()
    if s in {"refusal", "refuse"}:
        return "refusal"
    if s in {"hallucination", "hallucinate"}:
        return "hallucination"
    if s in {"unknown", "nan", "none", ""}:
        return "other"
    return s


def load_one_judged_csv(path: Path) -> pd.DataFrame:
    model_key, split = parse_model_key_and_split_from_filename(path)

    df = pd.read_csv(path)

    if "task" not in df.columns:
        raise ValueError(f"Missing 'task' column in {path}")

    label_col = None
    for cand in ["judge_label_norm", "judge_label", "lm_judge_label"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(f"Cannot find judge label column in {path}. Expected judge_label*.")

    df = df.copy()
    df["model_key"] = model_key
    df["split"] = split
    df["relation"] = df["task"].astype(str)
    df["judge_label_norm"] = df[label_col].map(normalize_label)

    df["judge_label_norm"] = df["judge_label_norm"].where(
        df["judge_label_norm"].isin(["refusal", "hallucination"]), other="other"
    )
    return df


def compute_behavior_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gb = df_all.groupby(["model_key", "relation"], dropna=False)
    for (model_key, relation), g in gb:
        total = int(len(g))
        refusal = int((g["judge_label_norm"] == "refusal").sum())
        halluc = int((g["judge_label_norm"] == "hallucination").sum())
        other = total - refusal - halluc

        refusal_rate = refusal / total if total else 0.0
        halluc_rate = halluc / total if total else 0.0

        r_low, r_high = wilson_ci(refusal, total)
        h_low, h_high = wilson_ci(halluc, total)

        rows.append(
            {
                "model_key": model_key,
                "relation": relation,
                "total": total,
                "refusal": refusal,
                "hallucination": halluc,
                "other": other,
                "refusal_rate": refusal_rate,
                "refusal_ci_low": r_low,
                "refusal_ci_high": r_high,
                "halluc_rate": halluc_rate,
                "halluc_ci_low": h_low,
                "halluc_ci_high": h_high,
            }
        )

    out = pd.DataFrame(rows)
    out["relation_order"] = out["relation"].map({r: i for i, r in enumerate(RELATION_ORDER)})
    out = out.sort_values(["model_key", "relation_order", "relation"], na_position="last").drop(
        columns=["relation_order"]
    )
    return out.reset_index(drop=True)


def load_lre_table(lre_dir: Path, model_keys: Iterable[str]) -> pd.DataFrame:
    all_rows = []
    for mk in sorted(set(model_keys)):
        f = lre_dir / f"{mk}.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"Missing LRE file for model_key={mk}: {f}\n"
                f"Expected one file per model in: {lre_dir}"
            )

        df = pd.read_csv(f)
        if "relation" not in df.columns:
            raise ValueError(f"LRE file missing 'relation' column: {f}")

        required = ["model_id", "num_layers", "subject_layer", "object_layer", "cos_improvement"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"LRE file {f} is missing required columns: {missing}")

        df = df.copy()
        df["model_key"] = mk

        keep = [
            "model_key",
            "model_id",
            "relation",
            "num_layers",
            "subject_layer",
            "object_layer",
            "n_total",
            "n_train",
            "n_test",
            "cos_mean",
            "cos_std",
            "base_cos_mean",
            "base_cos_std",
            "cos_improvement",
            "mse",
        ]
        keep_existing = [c for c in keep if c in df.columns]
        all_rows.append(df[keep_existing])

    return pd.concat(all_rows, ignore_index=True)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_plots(df_plus: pd.DataFrame, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key, g in df_plus.groupby("model_key"):
        g = g.copy()

        order_map = {r: i for i, r in enumerate(RELATION_ORDER)}
        g["relation_order"] = g["relation"].map(order_map).fillna(1e9)
        g = g.sort_values(["relation_order", "relation"]).drop(columns=["relation_order"])

        rels = g["relation"].tolist()
        y = np.arange(len(rels))

        halluc = g["halluc_rate"].to_numpy()
        refusal = g["refusal_rate"].to_numpy()

        halluc_err = np.vstack([
            halluc - g["halluc_ci_low"].to_numpy(),
            g["halluc_ci_high"].to_numpy() - halluc,
        ])
        refusal_err = np.vstack([
            refusal - g["refusal_ci_low"].to_numpy(),
            g["refusal_ci_high"].to_numpy() - refusal,
        ])

        fig = plt.figure(figsize=(12, 4.5))
        ax = plt.gca()

        ax.barh(y - 0.18, halluc * 100.0, height=0.35, xerr=halluc_err * 100.0, label="Hallucination")
        ax.barh(y + 0.18, refusal * 100.0, height=0.35, xerr=refusal_err * 100.0, label="Refusal")

        ax.set_yticks(y)
        ax.set_yticklabels(rels, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("Rate (%)")
        ax.set_title(f"{model_key}: refusal vs. hallucination")
        ax.set_xlim(0, 100)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

        fig.tight_layout()
        fig.savefig(out_dir / f"{model_key}_behavior_bars_{tag}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(7, 6))
        ax = plt.gca()

        x = g["cos_improvement"].to_numpy()
        yv = g["halluc_rate"].to_numpy()

        ax.scatter(x, yv)

        for _, row in g.iterrows():
            ax.text(float(row["cos_improvement"]) + 0.01, float(row["halluc_rate"]) + 0.01, str(row["relation"]))

        ax.set_xlabel("LRE cosine improvement (Δcos)")
        ax.set_ylabel("Hallucination rate")
        ax.set_title(f"{model_key}: linearity vs. hallucination")
        ax.set_xlim(0.30, 0.90)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_dir / f"{model_key}_lre_vs_halluc_scatter_{tag}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-dirs",
        nargs="+",
        required=True,
        help="One or more directories (or files) that contain *_with_judge.csv. "
             "The script searches recursively within directories.",
    )
    parser.add_argument(
        "--lre-dir",
        type=str,
        default=str(DEFAULT_LRE_DIR),
        help=f"Directory containing per-model LRE CSVs (default: {DEFAULT_LRE_DIR}). "
             "Must contain files like 'gemma_7b_it.csv'.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Where to write merged CSVs and plots.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag used in plot filenames. If empty, use out-dir basename.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not generate PNG figures.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lre_dir = Path(args.lre_dir)
    tag = args.tag.strip() or out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_with_judge_csvs(args.judge_dirs)
    if not files:
        raise RuntimeError(f"No *_with_judge.csv found under: {args.judge_dirs}")

    print("[post] Using LRE dir:", lre_dir.resolve())
    print("[post] Found judged CSVs:")
    for f in files:
        print("  -", f)

    dfs = [load_one_judged_csv(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)

    merged_path = out_dir / "all_with_judge_merged.csv"
    df_all.to_csv(merged_path, index=False)
    print(f"[post] Wrote merged judged rows: {merged_path} (n={len(df_all)})")

    df_sum = compute_behavior_summary(df_all)
    sum_path = out_dir / "behavior_summary_24rows.csv"
    df_sum.to_csv(sum_path, index=False)
    print(f"[post] Wrote behavior summary: {sum_path} (rows={len(df_sum)})")

    df_lre = load_lre_table(lre_dir=lre_dir, model_keys=df_sum["model_key"].unique())
    plus = df_sum.merge(df_lre, on=["model_key", "relation"], how="left", validate="one_to_one")

    missing = plus["cos_improvement"].isna().sum()
    if missing:
        bad = plus[plus["cos_improvement"].isna()][["model_key", "relation"]]
        raise RuntimeError(
            f"Join failed for {missing} (model_key, relation) pairs. "
            f"Check that LRE relations match eval tasks.\n{bad.to_string(index=False)}"
        )

    plus_path = out_dir / "behavior_plus_lre.csv"
    plus.to_csv(plus_path, index=False)
    print(f"[post] Wrote behavior+LRE: {plus_path} (rows={len(plus)})")

    pear_rows = []
    for mk, g in plus.groupby("model_key"):
        r = pearson_r(g["cos_improvement"].to_numpy(), g["halluc_rate"].to_numpy())
        pear_rows.append({"model_key": mk, "pearson_r_lre_vs_halluc": r})
    df_pear = pd.DataFrame(pear_rows).sort_values("model_key")
    pear_path = out_dir / "pearson_by_model.csv"
    df_pear.to_csv(pear_path, index=False)
    print(f"[post] Wrote Pearson-by-model: {pear_path}")

    if not args.no_plots:
        save_plots(plus, out_dir=out_dir, tag=tag)
        print(f"[post] Wrote plots into: {out_dir}")

    print("[post] Done.")


if __name__ == "__main__":
    main()
