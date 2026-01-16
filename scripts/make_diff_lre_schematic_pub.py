#!/usr/bin/env python3
"""
Polished schematic for difference-vector LRE (train/test), tuned for papers.

Fixes vs. previous versions:
  (1) \bar{d}_r is no longer "floating": d1,d2 visually feed into a mean() box.
  (2) All dashed/dotted arrows are forced to be straight (no accidental curvature).
  (3) "error" label placement avoids overlap with \hat{o}_j label.
  (4) Times-like serif typography + B/W print-friendly line styles.

Outputs:
  - diff_lre_schematic_pub_v5.pdf
  - diff_lre_schematic_pub_v5.png
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# -----------------------------
# Global style (Times-like)
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def add_point(ax, xy: Tuple[float, float], label: str, label_xy: Tuple[float, float],
              marker: str = "o") -> None:
    ax.scatter([xy[0]], [xy[1]], s=80, marker=marker,
               facecolors="white", edgecolors="black", linewidths=1.2, zorder=5)
    ax.text(label_xy[0], label_xy[1], label, fontsize=12, ha="center", va="center")


def add_arrow(ax, start: Tuple[float, float], end: Tuple[float, float],
              lw: float, ls: str,
              text: str | None = None, text_xy: Tuple[float, float] | None = None,
              text_size: float = 12,
              shrink: float = 6.0,
              z: int = 3) -> None:
    arrowprops = dict(
        arrowstyle="-|>",
        lw=lw,
        linestyle=ls,
        color="black",
        mutation_scale=14,
        shrinkA=shrink,
        shrinkB=shrink,
        connectionstyle="arc3,rad=0.0",  # enforce straight
    )
    ax.annotate("", xy=end, xytext=start, arrowprops=arrowprops, zorder=z)
    if text is not None and text_xy is not None:
        ax.text(text_xy[0], text_xy[1], text, fontsize=text_size, ha="center", va="center")


def add_box(ax, xy: Tuple[float, float], w: float, h: float, text: str,
            fontsize: float = 10.5) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="black",
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, fontsize=fontsize, ha="center", va="center")
    return box


def draw_train(ax) -> None:
    ax.set_title("Train (estimate relation direction)", fontsize=13, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Two training pairs
    s1, o1 = (0.14, 0.76), (0.42, 0.90)
    s2, o2 = (0.14, 0.38), (0.42, 0.52)

    add_point(ax, s1, r"$\mathbf{s}_1$", (s1[0], s1[1]-0.10))
    add_point(ax, o1, r"$\mathbf{o}_1$", (o1[0]+0.03, o1[1]+0.08))
    add_point(ax, s2, r"$\mathbf{s}_2$", (s2[0], s2[1]-0.10))
    add_point(ax, o2, r"$\mathbf{o}_2$", (o2[0]+0.03, o2[1]+0.08))

    # Difference vectors
    add_arrow(ax, s1, o1, lw=1.6, ls="--",
              text=r"$\mathbf{d}_1$", text_xy=(0.27, 0.88), text_size=12)
    add_arrow(ax, s2, o2, lw=1.6, ls="--",
              text=r"$\mathbf{d}_2$", text_xy=(0.27, 0.50), text_size=12)

    # Mean box: explicitly connects d_i -> \bar{d}_r
    box_xy = (0.24, 0.06)
    box_w, box_h = 0.46, 0.18
    box_text = r"$\bar{\mathbf{d}}_r=\frac{1}{|T|}\sum_{i\in T}\mathbf{d}_i$"
    box = add_box(ax, box_xy, box_w, box_h, box_text, fontsize=10.5)

    # Short connector arrows from d1/d2 into the box (keep them subtle)
    mid1 = ((s1[0]+o1[0])/2, (s1[1]+o1[1])/2)
    mid2 = ((s2[0]+o2[0])/2, (s2[1]+o2[1])/2)
    box_top_left  = (box_xy[0] + box_w*0.30, box_xy[1] + box_h)
    box_top_right = (box_xy[0] + box_w*0.70, box_xy[1] + box_h)
    add_arrow(ax, mid1, box_top_left,  lw=1.0, ls="-", shrink=2.0, z=2)
    add_arrow(ax, mid2, box_top_right, lw=1.0, ls="-", shrink=2.0, z=2)

    # Output direction vector \bar{d}_r (originating from the box)
    out_start = (box_xy[0] + box_w, box_xy[1] + box_h/2)
    out_end   = (0.92, 0.22)
    add_arrow(ax, out_start, out_end, lw=2.0, ls="-",
              text=r"$\bar{\mathbf{d}}_r$", text_xy=(0.82, 0.28), text_size=12)


def draw_test(ax) -> None:
    ax.set_title("Test (predict & evaluate)", fontsize=13, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Place points to avoid overlaps
    sj   = (0.14, 0.26)
    ohat = (0.56, 0.40)
    oj   = (0.82, 0.84)

    add_point(ax, sj,   r"$\mathbf{s}_j$", (sj[0], sj[1]-0.10), marker="o")
    add_point(ax, ohat, r"$\hat{\mathbf{o}}_j$", (ohat[0]+0.12, ohat[1]-0.02), marker="s")
    add_point(ax, oj,   r"$\mathbf{o}_j$", (oj[0]+0.06, oj[1]+0.08), marker="o")

    # Prediction: \hat{o}_j = s_j + \bar{d}_r
    add_arrow(ax, sj, ohat, lw=2.0, ls="-",
              text=r"$+\;\bar{\mathbf{d}}_r$", text_xy=(0.36, 0.30), text_size=12)

    # Baseline: s_j -> o_j
    add_arrow(ax, sj, oj, lw=1.6, ls="--")

    # Error: \hat{o}_j -> o_j (straight dotted)
    add_arrow(ax, ohat, oj, lw=1.6, ls=":",
              text=r"error", text_xy=(0.74, 0.60), text_size=11)

    # Compact metric definitions (no heavy box)
    txt = (
        r"$\cos^{\mathrm{base}}=\cos(\mathbf{s}_j,\mathbf{o}_j)$" "\n"
        r"$\cos^{\mathrm{lre}}=\cos(\hat{\mathbf{o}}_j,\mathbf{o}_j)$" "\n"
        r"$\Delta\cos=\mathbb{E}[\cos^{\mathrm{lre}}]-\mathbb{E}[\cos^{\mathrm{base}}]$"
    )
    ax.text(0.05, 0.95, txt, fontsize=10.5, ha="left", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")
    parser.add_argument("--basename", type=str, default="diff_lre_schematic_pub_v5",
                        help="Base filename without extension.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pdf_path = os.path.join(args.outdir, args.basename + ".pdf")
    png_path = os.path.join(args.outdir, args.basename + ".png")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    draw_train(axes[0])
    draw_test(axes[1])

    fig.tight_layout(pad=0.8, w_pad=1.0)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] Wrote:\n  {pdf_path}\n  {png_path}")


if __name__ == "__main__":
    main()
