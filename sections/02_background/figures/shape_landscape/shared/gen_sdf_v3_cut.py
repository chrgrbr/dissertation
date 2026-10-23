"""SDF v3 -- cutaway. Plane dissects the liver; near half (camera side) removed,
cut face opaque with the SDF field + bold zero-contour, far half solid behind for 3D context.
Run from repo root: python sections/02_background/figures/shape_landscape/shared/gen_sdf_v3_cut.py
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

er, ar = np.radians(ELEV), np.radians(AZIM)
cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])

sdf = ndi.distance_transform_edt(~b) - ndi.distance_transform_edt(b)
P = np.argwhere(b).astype(float); cen = P.mean(0)
w, V = np.linalg.eigh(np.cov((P - cen).T)); order = w.argsort()[::-1]
v1, v2 = V[:, order[0]], V[:, order[1]]          # slice plane spanned by 2 longest axes
n = V[:, order[2]]                                # plane normal (shortest axis)
if n @ cam < 0:
    n = -n                                        # orient toward camera

# cut face field on the plane
p1, p2 = (P - cen) @ v1, (P - cen) @ v2
s = np.linspace(p1.min() - 2, p1.max() + 2, 160); t = np.linspace(p2.min() - 2, p2.max() + 2, 160)
S, T = np.meshgrid(s, t, indexing="ij")
coords = cen[None, None, :] + S[..., None] * v1 + T[..., None] * v2
C = ndi.map_coordinates(sdf, [coords[..., 0], coords[..., 1], coords[..., 2]],
                        order=1, mode="constant", cval=sdf.max())
vmax = max(-C.min(), 1.0) * 1.2; norm = colors.Normalize(-vmax, vmax); cmap = cm.RdBu

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection="3d")

# far half only: keep triangles behind the plane (drop camera-side / near half)
cents = tri.mean(axis=1)
keep = (cents - cen) @ n <= 0
mfar = Poly3DCollection(tri[keep], alpha=1.0)
mfar.set_facecolor("#aebfd2"); mfar.set_edgecolor("#6f8aa8"); mfar.set_linewidth(0.15)
ax.add_collection3d(mfar)

# opaque cut face on top
ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2], facecolors=cmap(norm(C)),
                shade=False, rstride=1, cstride=1, antialiased=False)
ds, dt = s[1] - s[0], t[1] - t[0]
eps = 1.0 * n                                     # slight lift toward camera
for cn in measure.find_contours(C, 0.0):
    pc3 = cen + (s[0] + cn[:, 0] * ds)[:, None] * v1 + (t[0] + cn[:, 1] * dt)[:, None] * v2 + eps
    ax.plot(pc3[:, 0], pc3[:, 1], pc3[:, 2], color="k", lw=2.2, zorder=100)

ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
ax.view_init(elev=ELEV, azim=AZIM); ax.set_box_aspect((nx, ny, nz)); ax.set_axis_off()
out = os.path.join(SL, "representations/4_sdf/versions/v3_cut_render.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
plt.close(fig); print("wrote", out)
