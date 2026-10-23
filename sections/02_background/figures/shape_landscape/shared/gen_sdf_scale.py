"""Colorbar/scale for the SDF tile: red = -inf (inside) -> white = 0 (surface) -> blue = +inf
(outside), matching the RdBu slice. Renders horizontal and vertical variants (transparent bg).
Run from repo root:
python sections/02_background/figures/shape_landscape/shared/gen_sdf_scale.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SL = "sections/02_background/figures/shape_landscape"
grad = np.linspace(0, 1, 256)[None, :]            # 0->red (RdBu low), 1->blue (RdBu high)


def horizontal():
    fig, ax = plt.subplots(figsize=(3.2, 0.62))
    ax.imshow(grad, aspect="auto", cmap="RdBu", extent=[0, 1, 0, 1])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("k")
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels([r"$-\infty$", r"$0$", r"$+\infty$"], fontsize=11)
    ax.tick_params(length=0, pad=3)
    ax.text(0.0, 1.35, "inside", ha="left", va="bottom", fontsize=8, color="0.3")
    ax.text(0.5, 1.35, "surface", ha="center", va="bottom", fontsize=8, color="0.3")
    ax.text(1.0, 1.35, "outside", ha="right", va="bottom", fontsize=8, color="0.3")
    out = os.path.join(SL, "representations/4_sdf/versions/sdf_scale_h.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig); print("wrote", out)


def vertical():
    fig, ax = plt.subplots(figsize=(0.72, 3.0))
    ax.imshow(grad.T[::-1], aspect="auto", cmap="RdBu", extent=[0, 1, 0, 1])  # top=+inf(blue)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("k")
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([r"$-\infty$", r"$0$", r"$+\infty$"], fontsize=11)
    ax.tick_params(length=0, pad=3)
    out = os.path.join(SL, "representations/4_sdf/versions/sdf_scale_v.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig); print("wrote", out)


horizontal()
vertical()
