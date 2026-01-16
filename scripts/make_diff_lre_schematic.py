#!/usr/bin/env python3
"""Generate a clean schematic for difference-vector LRE (train vs test).

Outputs:
  - PDF (vector) for papers
  - PNG (raster) for quick viewing

This script is deterministic and uses manual label offsets + small white
label boxes to avoid overlaps in a 2-panel diagram.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def _setup_matplotlib():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "pdf.fonttype": 42,  # TrueType in PDF
            "ps.fonttype": 42,
        }
    )


def draw_arrow(ax, start, end, *, lw=2.0, ls="-", ms=14, z=2, rad=0.0):
    """Draw a nice arrow from start -> end."""
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color="black",
        shrinkA=0.0,
        shrinkB=0.0,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(arrow)
    return arrow


def annotate(ax, xy, text, dx, dy, *, ha="left", va="bottom", fontsize=11):
    """Place a label with a small white box to keep text readable."""
    ax.annotate(
        text,
        xy=xy,
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.90),
        zorder=10,
    )


def label_mid(ax, start, end, text, *, dy=0, fontsize=11, t=0.55):
    """Label a segment near its midpoint."""
    x = start[0] * (1 - t) + end[0] * t
    y = start[1] * (1 - t) + end[1] * t
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.90),
        zorder=10,
    )


def make_figure(out_pdf: str, out_png: str) -> None:
    _setup_matplotlib()

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.0, 3.0), gridspec_kw={"wspace": 0.15}
    )

    for ax in (axL, axR):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # Shared (schematic) direction for the relation in 2D
    d = np.array([2.0, 0.8])

    # ----------------
    # Left: Train
    # ----------------
    s1 = np.array([0.8, 0.9])
    o1 = s1 + d
    s2 = np.array([0.9, 2.1])
    o2 = s2 + d

    # Mean direction shown separately to reduce overlaps
    d_start = np.array([0.6, 0.25])
    d_end = d_start + d

    # Points
    axL.scatter(
        [s1[0], s2[0]],
        [s1[1], s2[1]],
        s=90,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=5,
    )
    axL.scatter(
        [o1[0], o2[0]],
        [o1[1], o2[1]],
        s=90,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=5,
    )

    # Per-example differences (dashed)
    draw_arrow(axL, s1, o1, lw=2.0, ls="--", ms=14, z=2)
    draw_arrow(axL, s2, o2, lw=2.0, ls="--", ms=14, z=2)

    # Mean direction (solid, thicker)
    draw_arrow(axL, d_start, d_end, lw=2.8, ls="-", ms=16, z=3)

    # Labels
    annotate(axL, s1, r"$\mathbf{s}_1$", dx=-12, dy=-12, ha="right", va="top")
    annotate(axL, o1, r"$\mathbf{o}_1$", dx=10, dy=4, ha="left", va="bottom")
    annotate(axL, s2, r"$\mathbf{s}_2$", dx=-12, dy=6, ha="right", va="bottom")
    annotate(axL, o2, r"$\mathbf{o}_2$", dx=10, dy=6, ha="left", va="bottom")

    label_mid(axL, s1, o1, r"$\mathbf{d}_1$", dy=16, fontsize=11, t=0.52)
    label_mid(axL, s2, o2, r"$\mathbf{d}_2$", dy=16, fontsize=11, t=0.52)
    label_mid(axL, d_start, d_end, r"$\bar{\mathbf{d}}_r$", dy=-16, fontsize=11, t=0.55)

    axL.set_xlim(0.0, 3.6)
    axL.set_ylim(0.0, 3.2)
    axL.set_title(r"Train: estimate $\bar{\mathbf{d}}_r$", pad=6)

    # ----------------
    # Right: Test
    # ----------------
    sj = np.array([0.9, 1.1])
    ohat = sj + d
    # Move the true object upward so the dashed arrow is visible (not hidden by the solid one)
    oj = ohat + np.array([0.15, 0.55])

    axR.scatter(
        [sj[0]],
        [sj[1]],
        s=95,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=5,
    )
    axR.scatter(
        [ohat[0]],
        [ohat[1]],
        s=95,
        facecolor="#e6e6e6",  # light gray for prediction
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )
    axR.scatter(
        [oj[0]],
        [oj[1]],
        s=95,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=5,
    )

    # Solid: predicted translation
    draw_arrow(axR, sj, ohat, lw=2.8, ls="-", ms=16, z=3)
    # Dashed: true subject->object relation (curved so it stays visible)
    draw_arrow(axR, sj, oj, lw=2.0, ls="--", ms=14, z=2, rad=0.12)
    # Dotted: prediction error
    draw_arrow(axR, ohat, oj, lw=1.6, ls=":", ms=12, z=2, rad=-0.25)

    annotate(axR, sj, r"$\mathbf{s}_j$", dx=-12, dy=-12, ha="right", va="top")
    annotate(axR, ohat, r"$\hat{\mathbf{o}}_j$", dx=10, dy=-8, ha="left", va="top")
    annotate(axR, oj, r"$\mathbf{o}_j$", dx=10, dy=6, ha="left", va="bottom")

    label_mid(axR, sj, ohat, r"$+\bar{\mathbf{d}}_r$", dy=16, fontsize=11, t=0.55)
    label_mid(axR, ohat, oj, r"error", dy=-14, fontsize=10, t=0.55)

    axR.set_xlim(0.0, 4.3)
    axR.set_ylim(0.0, 3.2)
    axR.set_title(r"Test: translate and score $\Delta\cos$", pad=6)

    # Tiny legend box (kept inside the panel)
    axR.text(
        0.02,
        0.02,
        "solid: prediction\n-- : target\n: : error",
        transform=axR.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="black", lw=0.6, alpha=0.90),
    )

    fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.08, wspace=0.18)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-pdf", required=True, help="Output PDF path (vector).")
    parser.add_argument("--out-png", required=True, help="Output PNG path (raster).")
    args = parser.parse_args()
    make_figure(args.out_pdf, args.out_png)
    print(f"[ok] wrote: {args.out_pdf}")
    print(f"[ok] wrote: {args.out_png}")


if __name__ == "__main__":
    main()
