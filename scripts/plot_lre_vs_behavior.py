import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


INPUT_PATH = "data/lre/natural_lre_vs_behavior.csv"
OUT_DIR = Path("experiments/figures")


def _order_relations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put relations in a consistent, interpretable order.
    Top of bar chart will be 'father'.
    """
    order = ["father", "instrument", "sport", "company_ceo"]
    df = df.copy()
    df["relation"] = pd.Categorical(df["relation"], categories=order, ordered=True)
    return df.sort_values("relation", ascending=False)


def plot_behavior_bars(df: pd.DataFrame, model_key: str, out_path: Path) -> None:
    """
    Horizontal bar chart: non-hallucination vs. hallucination (with CIs).
    Legend placed below the axes to avoid overlapping with title / bars.
    """
    sub = df[df["model_key"] == model_key].copy()
    sub = _order_relations(sub)

    relations = sub["relation"].tolist()
    y_pos = np.arange(len(relations))

    # rates in %
    hall = sub["halluc_rate"].to_numpy() * 100.0
    refu = sub["refusal_rate"].to_numpy() * 100.0

    # CI half-widths ( (high - low) / 2 ), also in %
    hall_err = (sub["halluc_ci_high"] - sub["halluc_ci_low"]).to_numpy() * 100.0 / 2.0
    refu_err = (sub["refusal_ci_high"] - sub["refusal_ci_low"]).to_numpy() * 100.0 / 2.0

    fig, ax = plt.subplots(figsize=(8.0, 3.2))

    bar_height = 0.35

    # hallucination bars
    ax.barh(
        y_pos + bar_height / 2.0,
        hall,
        height=bar_height,
        xerr=hall_err,
        error_kw=dict(capsize=2, linewidth=0.8),
        label="Hallucination",
    )

    # refusal bars (distinguished by hatch only; no explicit color)
    ax.barh(
        y_pos - bar_height / 2.0,
        refu,
        height=bar_height,
        xerr=refu_err,
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

    # Figure-level legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )

    # Leave room at bottom for legend
    plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lre_scatter(df: pd.DataFrame, model_key: str, out_path: Path) -> None:
    """
    Scatter: LRE cosine improvement (x) vs. hallucination rate (y).

    All labels are placed to the *right* of their points with a small
    horizontal offset in points. This avoids collisions with the point,
    axes, and grid lines, and keeps the layout visually clean.
    """
    sub = df[df["model_key"] == model_key].copy()
    sub = _order_relations(sub)

    x = sub["cos_improvement"].to_numpy()
    y = sub["halluc_rate"].to_numpy()
    labels = sub["relation"].tolist()

    fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
    ax.scatter(x, y)

    # Fixed offset in display (points) to the right of each point
    for xx, yy, label in zip(x, y, labels):
        ax.annotate(
            label,
            xy=(xx, yy),
            xytext=(6, 0),              # 6 points to the right
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("LRE cosine improvement (Δcos)")
    ax.set_ylabel("Hallucination rate")

    # Small margins so labels are not clipped at the borders
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.4, 1.0)

    ax.set_xticks(np.linspace(0.4, 1.0, 4))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_title(f"{model_key}: linearity vs. hallucination")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    model_keys = sorted(df["model_key"].unique())

    for mk in model_keys:
        print(f"[plot] {mk}")

        behavior_path = OUT_DIR / f"{mk}_behavior_bars.png"
        scatter_path = OUT_DIR / f"{mk}_lre_vs_halluc_scatter.png"

        plot_behavior_bars(df, mk, behavior_path)
        plot_lre_scatter(df, mk, scatter_path)


if __name__ == "__main__":
    main()
