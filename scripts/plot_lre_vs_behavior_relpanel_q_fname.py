import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _order_relations(df: pd.DataFrame) -> pd.DataFrame:
    order = ["father", "instrument", "sport", "company_ceo", "company_hq", "country_language"]
    df = df.copy()
    df["relation"] = pd.Categorical(df["relation"], categories=order, ordered=True)
    return df.sort_values("relation", ascending=False)


def plot_behavior_bars(df: pd.DataFrame, model_key: str, out_path: Path) -> None:
    sub = df[df["model_key"] == model_key].copy()
    sub = _order_relations(sub)

    relations = sub["relation"].astype(str).tolist()
    y_pos = np.arange(len(relations))

    hall = sub["halluc_rate"].to_numpy() * 100.0
    nonhall = sub["refusal_rate"].to_numpy() * 100.0

    hall_err = (sub["halluc_ci_high"] - sub["halluc_ci_low"]).to_numpy() * 100.0 / 2.0
    nonhall_err = (sub["refusal_ci_high"] - sub["refusal_ci_low"]).to_numpy() * 100.0 / 2.0

    fig, ax = plt.subplots(figsize=(8.2, 3.6))

    bar_height = 0.34

    ax.barh(
        y_pos + bar_height / 2.0,
        hall,
        height=bar_height,
        xerr=hall_err,
        error_kw=dict(capsize=2, linewidth=0.8),
        label="Hallucination",
    )

    ax.barh(
        y_pos - bar_height / 2.0,
        nonhall,
        height=bar_height,
        xerr=nonhall_err,
        hatch="//",
        error_kw=dict(capsize=2, linewidth=0.8),
        label="Non-hallucination",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(relations)
    ax.set_xlabel("Rate (%)")
    ax.set_xlim(0, 100)

    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_title(f"{model_key}: non-hallucination vs. hallucination", pad=10)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_label_offsets(y: np.ndarray, y_thresh: float = 0.03) -> np.ndarray:
    """
    Assign small vertical offsets (in points) to reduce overlaps for nearby y values.
    Labels remain to the right of points.
    """
    n = len(y)
    dy = np.zeros(n, dtype=int)
    idx = np.argsort(y)

    groups = []
    cur = [idx[0]]
    for j in idx[1:]:
        if abs(y[j] - y[cur[-1]]) <= y_thresh:
            cur.append(j)
        else:
            groups.append(cur)
            cur = [j]
    groups.append(cur)

    for g in groups:
        if len(g) <= 1:
            continue
        # Spread within the group
        offs = np.linspace(-12, 12, len(g))
        for k, o in zip(g, offs):
            dy[k] = int(round(o))
    return dy


def plot_lre_scatter(df: pd.DataFrame, model_key: str, out_path: Path, xlim=None) -> None:
    sub = df[df["model_key"] == model_key].copy()
    sub = _order_relations(sub)

    x = sub["cos_improvement"].to_numpy(dtype=float)
    y = sub["halluc_rate"].to_numpy(dtype=float)
    labels = sub["relation"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(4.4, 4.0), constrained_layout=True)
    ax.scatter(x, y)

    dy = _compute_label_offsets(y, y_thresh=0.03)

    for xx, yy, label, dyy in zip(x, y, labels, dy):
        ax.annotate(
            label,
            xy=(xx, yy),
            xytext=(6, int(dyy)),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("LRE cosine improvement (Δcos)")
    ax.set_ylabel("Hallucination rate")

    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    if xlim is not None:
        ax.set_xlim(*xlim)
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_title(f"{model_key}: linearity vs. hallucination")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Joined CSV (output of join step).")
    ap.add_argument("--out-dir", required=True, help="Directory to write PNGs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    model_keys = sorted(df["model_key"].unique().tolist())

    # Use global x-limits for comparability across models
    x_min = float(np.floor((df["cos_improvement"].min() - 0.05) * 10) / 10)
    x_max = float(np.ceil((df["cos_improvement"].max() + 0.05) * 10) / 10)
    xlim = (x_min, x_max)

    for mk in model_keys:
        print(f"[plot] {mk}")
        plot_behavior_bars(df, mk, out_dir / f"{mk}_behavior_bars.png")
        plot_lre_scatter(df, mk, out_dir / f"{mk}_lre_vs_halluc_scatter.png", xlim=xlim)

    print("[done] figures in:", out_dir)


if __name__ == "__main__":
    main()
