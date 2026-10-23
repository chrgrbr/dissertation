"""Reduce one liver segmentation to several shape representations and render individual PNGs.
Data: abdomen_mrct fixed_seg, label 1 = liver (canonical RAS).
Outputs: one PNG per representation in representations/N_*/versions/v1_render.png
View locked to elev=15, azim=-120 for all 3D panels (incl. the 3D-embedded SDF slice).
"""
import os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
ELEV, AZIM = 15, -120
ACCENT, FACE, EDGE = "#3b6ea5", "#bcd0e6", "#3b6ea5"
rng = np.random.default_rng(0)

# Base output directory
SL = "sections/02_background/figures/shape_landscape"

img = nib.as_closest_canonical(nib.load(SEG))
d = np.asarray(img.dataobj); liver = (d == 1)
xs, ys, zs = np.where(liver)
sl = tuple(slice(max(c.min() - 2, 0), c.max() + 3) for c in (xs, ys, zs))
liver = liver[sl].astype(float)
nx, ny, nz = liver.shape
b = liver > 0.5
verts, faces, _, _ = measure.marching_cubes(ndi.gaussian_filter(liver, 1.0), level=0.5, step_size=2)
tri = verts[faces]

def box(ax):
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.view_init(elev=ELEV, azim=AZIM); ax.set_box_aspect((nx, ny, nz)); ax.set_axis_off()

def save(fig, rep_folder, filename="v1_render.png"):
    out_dir = os.path.join(SL, "representations", rep_folder, "versions")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, filename)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print("wrote", out)

# camera (to-camera) unit vector for this view; used to orient the SDF plane and depth-sort landmarks
er, ar = np.radians(ELEV), np.radians(AZIM)
cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])

# 5) binary / voxel mask
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
f = 4; small = b[::f, ::f, ::f]
ax.voxels(small, facecolors=ACCENT + "cc", edgecolor="#234", linewidth=0.2)
ax.view_init(elev=ELEV, azim=AZIM); ax.set_box_aspect(small.shape); ax.set_axis_off()
save(fig, "5_binary_mask")

# 2) point cloud
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
surf = b & ~ndi.binary_erosion(b); pts = np.argwhere(surf)
pc = pts[rng.choice(len(pts), size=min(2500, len(pts)), replace=False)]
ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2, c=ACCENT, depthshade=True)
box(ax)
save(fig, "2_point_cloud")

# 3) mesh / surface
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
m = Poly3DCollection(tri, alpha=0.9); m.set_facecolor(FACE); m.set_edgecolor(EDGE); m.set_linewidth(0.12)
ax.add_collection3d(m); box(ax)
save(fig, "3_mesh")

# 4) signed distance function -- colored slice on the liver's MAXIMAL cross-section (PCA plane), in 3D
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
sdf = ndi.distance_transform_edt(~b) - ndi.distance_transform_edt(b)
P = np.argwhere(b).astype(float); cen = P.mean(0)
w, V = np.linalg.eigh(np.cov((P - cen).T)); order = w.argsort()[::-1]
v1, v2 = V[:, order[0]], V[:, order[1]]               # longest, second axis -> plane they span
p1, p2 = (P - cen) @ v1, (P - cen) @ v2
s = np.linspace(p1.min() - 4, p1.max() + 4, 160); t = np.linspace(p2.min() - 4, p2.max() + 4, 160)
S, T = np.meshgrid(s, t, indexing="ij")
coords = cen[None, None, :] + S[..., None] * v1 + T[..., None] * v2
C = ndi.map_coordinates(sdf, [coords[..., 0], coords[..., 1], coords[..., 2]],
                        order=1, mode="constant", cval=sdf.max())
vmax = max(-C.min(), 1.0) * 1.2; norm = colors.Normalize(-vmax, vmax); cmap = cm.RdBu  # clamp -> vivid boundary
mc = Poly3DCollection(tri, alpha=0.10); mc.set_facecolor("#999"); mc.set_edgecolor("none")
ax.add_collection3d(mc)
ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2], facecolors=cmap(norm(C)),
                shade=False, rstride=1, cstride=1, antialiased=False)
ds, dt = s[1] - s[0], t[1] - t[0]
for cn in measure.find_contours(C, 0.0):
    pc3 = cen + (s[0] + cn[:, 0] * ds)[:, None] * v1 + (t[0] + cn[:, 1] * dt)[:, None] * v2
    ax.plot(pc3[:, 0], pc3[:, 1], pc3[:, 2], color="k", lw=1.2)
box(ax)
save(fig, "4_sdf")

# 1) landmarks -- farthest-point-sampled; near side solid, occluded (far) side hollow
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")
# Need pts from surface (recompute if needed — pts was overwritten by scatter above)
surf2 = b & ~ndi.binary_erosion(b); pts2 = np.argwhere(surf2)
m2 = Poly3DCollection(tri, alpha=0.20); m2.set_facecolor("#aaa"); m2.set_edgecolor("none")
ax.add_collection3d(m2)
def fps(P, k):
    idx = [rng.integers(len(P))]; dist = np.full(len(P), np.inf)
    for _ in range(k - 1):
        dist = np.minimum(dist, np.linalg.norm(P - P[idx[-1]], axis=1)); idx.append(int(dist.argmax()))
    return P[idx]
lms = fps(pts2.astype(float), 7)
depth = (lms - pts2.mean(0)) @ cam            # >=0 near side, <0 occluded
near, far = lms[depth >= 0], lms[depth < 0]
ax.scatter(far[:, 0], far[:, 1], far[:, 2], s=45, facecolors="none", edgecolors="#cc3333",
           linewidths=1.5, depthshade=False)
ax.scatter(near[:, 0], near[:, 1], near[:, 2], s=50, c="#cc3333", depthshade=False)
box(ax)
save(fig, "1_landmarks")
