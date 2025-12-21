import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = "data/lre/ext6/natural_lre_vs_behavior.csv"
OUT_DIR = Path("experiments/figures_ext6")

RELATION_ORDER = ["father", "instrument", "sport", "company_ceo", "country_language", "company_hq"]


def _order_relations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["relation"] = pd.Categorical(df["relation"], categories=RELATION_ORDER, ordered=True)
    # Sort descending so that 'father' ends up at the top of the horizontal bar chart
    return df.sort_values("relation", ascending=False)


def plot_behavior_bars(df: pd.DataFrame, model_key: str, out_path: Path) -> None:
    sub = df[df["model_key"] == model_key].copy()
    sub = _order_relations(sub)

    relations = sub["relation"].tolist()
    y_pos = np.arange(len(relations))

    hall = sub["halluc_rate"].to_numpy() * 100.0
    nonhall = sub["refusal_rate"].to_numpy() * 100.0

    hall_err = (sub["halluc_ci_high"] - sub["halluc_ci_low"]).to_numpy() * 100.0 / 2.0
    nonhall_err = (sub["refusal_ci_high"] - sub["refusal_ci_low"]).to_numpy() * 100.0 / 2.0

    fig_h = 1.0 + 0.5 * len(relations)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))

    bar_height = 0.35

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


def plot_lre_scatter(df: pd.DataFrame, model_key: str, out_path: Path) -> None:
    sub = df[df["model_key"] == model_key].copy()
    sub = sub[sub["relation"].isin(RELATION_ORDER)].copy()

    x = sub["cos_improvement"].to_numpy()
    y = sub["halluc_rate"].to_numpy()
    labels = sub["relation"].tolist()

    fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
    ax.scatter(x, y)

    # Label placement: right of the point, with small vertical offsets for close y-values
    order = np.argsort(y)
    used = []
    threshold = 0.05
    dy_cycle = [0, 8, -8, 16, -16, 24, -24]

    for idx in order:
        xx, yy, label = x[idx], y[idx], labels[idx]

        dy = 0
        if used:
            if any(abs(yy - prev_y) < threshold for prev_y, _ in used):
                dy = dy_cycle[min(len(used), len(dy_cycle) - 1)]
        used.append((yy, dy))

        ax.annotate(
            label,
            xy=(xx, yy),
            xytext=(6, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("LRE cosine improvement (Δcos)")
    ax.set_ylabel("Hallucination rate")

    # Dynamic x-limits with padding, clamped to [0, 1]
    xmin = max(0.0, float(np.min(x)) - 0.05)
    xmax = min(1.0, float(np.max(x)) + 0.05)
    if xmax - xmin < 0.2:
        mid = 0.5 * (xmin + xmax)
        xmin = max(0.0, mid - 0.1)
        xmax = min(1.0, mid + 0.1)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.05, 1.05)

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_title(f"{model_key}: linearity vs. hallucination")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)
    model_keys = sorted(df["model_key"].unique())

    for mk in model_keys:
        print(f"[plot] {mk}")
        plot_behavior_bars(df, mk, OUT_DIR / f"{mk}_behavior_bars.png")
        plot_lre_scatter(df, mk, OUT_DIR / f"{mk}_lre_vs_halluc_scatter.png")


if __name__ == "__main__":
    main()
