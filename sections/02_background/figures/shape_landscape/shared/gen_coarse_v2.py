"""Coarser v2 renders of point cloud, mesh, binary mask -- better visible in print.
Same liver, same locked view/colors as gen_representations.py. Saves versions/v2_render.png.
Run from repo root: python sections/02_background/figures/shape_landscape/shared/gen_coarse_v2.py
"""
import os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
ELEV, AZIM = 15, -120
ACCENT, FACE, EDGE = "#3b6ea5", "#bcd0e6", "#3b6ea5"
rng = np.random.default_rng(0)
SL = "sections/02_background/figures/shape_landscape"

img = nib.as_closest_canonical(nib.load(SEG))
d = np.asarray(img.dataobj); liver = (d == 1)
xs, ys, zs = np.where(liver)
sl = tuple(slice(max(c.min() - 2, 0), c.max() + 3) for c in (xs, ys, zs))
liver = liver[sl].astype(float)
nx, ny, nz = liver.shape
b = liver > 0.5

def box(ax):
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.view_init(elev=ELEV, azim=AZIM); ax.set_box_aspect((nx, ny, nz)); ax.set_axis_off()

def save(fig, rep_folder, filename="v2_render.png"):
    out_dir = os.path.join(SL, "representations", rep_folder, "versions")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, filename)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print("wrote", out)

# 5) binary mask -- coarser voxels (f=4 -> f=7)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
f = 7; small = b[::f, ::f, ::f]
ax.voxels(small, facecolors=ACCENT + "cc", edgecolor="#234", linewidth=0.4)
ax.view_init(elev=ELEV, azim=AZIM); ax.set_box_aspect(small.shape); ax.set_axis_off()
save(fig, "5_binary_mask")

# 2) point cloud -- fewer, larger dots (2500@s=2 -> 500@s=10)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
surf = b & ~ndi.binary_erosion(b); pts = np.argwhere(surf)
pc = pts[rng.choice(len(pts), size=min(500, len(pts)), replace=False)]
ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=10, c=ACCENT, depthshade=True)
box(ax)
save(fig, "2_point_cloud")

# 3) mesh / surface -- coarser triangles (step_size 2 -> 5), thicker edges
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
verts, faces, _, _ = measure.marching_cubes(ndi.gaussian_filter(liver, 1.2), level=0.5, step_size=5)
tri = verts[faces]
m = Poly3DCollection(tri, alpha=0.9); m.set_facecolor(FACE); m.set_edgecolor(EDGE); m.set_linewidth(0.5)
ax.add_collection3d(m); box(ax)
save(fig, "3_mesh")
