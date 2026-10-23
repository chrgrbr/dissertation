"""SDF v6 -- flat icon built from the SHARED liver silhouette (\\liveroutline), so the SDF
tile is visually comparable to the learned icons (same liver shape + lavender palette).
The signed-distance field is shown as nested iso-distance rings + a bold zero-level outline.
Run from repo root:
python sections/02_background/figures/shape_landscape/shared/gen_sdf_v6.py
"""
import os, re
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.path import Path

SL = "sections/02_background/figures/shape_landscape"
OUTLINE = os.path.join(SL, "shared/liver_outline.tex")

# --- parse the shared \liveroutline coordinates (normalised, width ~= 1) ---
txt = open(OUTLINE).read()
pts = re.findall(r"\(([-\d.]+),([-\d.]+)\)", txt)
poly = np.array([[float(x), float(y)] for x, y in pts])

# --- rasterise the silhouette on a fine grid, with margin for the exterior field ---
R = 600
mrg = 0.55                                   # exterior margin (in silhouette-width units)
lo = poly.min(0) - mrg; hi = poly.max(0) + mrg
xs = np.linspace(lo[0], hi[0], R); ys = np.linspace(lo[1], hi[1], R)
X, Y = np.meshgrid(xs, ys)
inside = Path(poly).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(R, R)

# --- signed distance (negative inside), in pixel units then normalised to width ---
px = xs[1] - xs[0]
sdf = (ndi.distance_transform_edt(~inside) - ndi.distance_transform_edt(inside)) * px

LAV = "#e8eaf6"        # pale lavender interior (tile-like)
INK = "#3b6ea5"        # accent outline / cool rings


def render(tag, diverging):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect("equal"); ax.axis("off")

    band = np.abs(sdf).max()
    if diverging:
        # soft warm interior -> lavender exterior, sign encoded by hue
        norm = colors.Normalize(-band * 0.9, band * 0.9)
        cmap = colors.LinearSegmentedColormap.from_list(
            "sdf", ["#c98b8b", "#f0d9d9", "#f4f4fb", "#dfe3f5", "#b9c2e8"])
        ax.pcolormesh(X, Y, norm(sdf), cmap=cmap, shading="gouraud", zorder=0)
    else:
        # monochrome lavender fill of the interior only
        fillrgba = np.zeros((R, R, 4))
        fillrgba[..., :3] = colors.to_rgb(LAV)
        fillrgba[..., 3] = np.where(inside, 1.0, 0.0)
        ax.imshow(fillrgba, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], zorder=0)

    # nested iso-distance rings (offsets of the boundary), thin
    step = band / 4.5
    for k in range(1, 4):
        for lvl, col in [(-k * step, "#7d4b4b" if diverging else INK),
                         (k * step, "#5b6bb0" if diverging else INK)]:
            for cn in measure.find_contours(sdf, lvl):
                ax.plot(xs[0] + cn[:, 1] * px, ys[0] + cn[:, 0] * px,
                        color=col, lw=0.7, alpha=0.55, zorder=2)

    # bold zero-level set = the liver outline itself
    ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
            color="#1f3a5f", lw=2.6, zorder=3)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    out = os.path.join(SL, "representations/4_sdf/versions", f"v6_{tag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig); print("wrote", out)


render("mono", diverging=False)
render("div", diverging=True)
